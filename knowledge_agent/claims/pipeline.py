"""Claim PDF and document-folder ingestion orchestration."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from knowledge_agent.claims.chunking import chunk_documents
from knowledge_agent.agents.document_classifier import (
    DocumentClassification,
    LogicalDocument,
    PageBoundaryDecision,
)
from knowledge_agent.claims.embeddings import TextEmbedder
from knowledge_agent.claims.filesystem import (
    ensure_claim_dirs,
    preserve_document_pdf,
    preserve_original_pdf,
    write_claim_manifest,
    write_json,
    write_jsonl,
)
from knowledge_agent.claims.ocr import OcrClient
from knowledge_agent.claims.models import (
    ClaimManifest,
    DocumentChunk,
    DocumentMetadata,
    PageRange,
    PageText,
    RetrievalMode,
    page_id_for,
    utc_now,
)
from knowledge_agent.claims.split import group_logical_documents, write_split_pdfs
from knowledge_agent.claims.vector_store import VectorStore


LOGGER = logging.getLogger(__name__)
_MAX_CONCURRENT_WORKERS = 4


@dataclass
class IngestionServices:
    ocr_client: OcrClient
    classify_document: Callable[[str, list[PageText]], DocumentClassification]
    classify_page_boundary: Callable[
        [PageText, PageText | None, LogicalDocument | None],
        PageBoundaryDecision,
    ]
    extract_document_metadata: Callable[
        [LogicalDocument, list[DocumentChunk]],
        DocumentMetadata,
    ]
    embedder: TextEmbedder | None
    vector_store_factory: Callable[[Path], VectorStore] | None
    retrieval_mode: RetrievalMode
    additional_retrieval_modes: tuple[RetrievalMode, ...] = ()
    lightrag_indexer: Callable[[Path, list[DocumentChunk]], object] | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None


class IngestionLogEntry(BaseModel):
    step: str
    status: str
    message: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class IngestionLog(BaseModel):
    claim_id: str
    entries: list[IngestionLogEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


@dataclass(frozen=True)
class ClassifiedSourceDocument:
    path: Path
    pages: list[PageText]
    classification: DocumentClassification


def ingest_claim_pdf(
    claim_id: str,
    pdf_path: Path,
    data_root: Path,
    services: IngestionServices,
) -> ClaimManifest:
    log = IngestionLog(claim_id=claim_id)
    root = ensure_claim_dirs(data_root, claim_id)

    with log_step(log, "preserve_original", root):
        original_pdf = preserve_original_pdf(pdf_path, root)

    with log_step(log, "ocr", root):
        pages = services.ocr_client.extract_pages(claim_id, original_pdf)
        _write_pages(root, pages)

    with log_step(log, "split", root):
        logical_documents = group_logical_documents(
            claim_id,
            pages,
            services.classify_page_boundary,
        )
        logical_documents = write_split_pdfs(
            original_pdf,
            logical_documents,
            root / "documents",
        )

    return _complete_ingestion(
        claim_id=claim_id,
        root=root,
        source_files=[original_pdf],
        logical_documents=logical_documents,
        services=services,
        log=log,
        locked_document_types=False,
    )


def ingest_claim_folder(
    claim_id: str,
    folder_path: Path,
    data_root: Path,
    services: IngestionServices,
) -> ClaimManifest:
    log = IngestionLog(claim_id=claim_id)
    root = ensure_claim_dirs(data_root, claim_id)

    with log_step(log, "collect_documents", root):
        pdf_paths = _collect_pdf_paths(folder_path)

    with log_step(log, "ocr_and_classify", root):
        def classify_source(pdf_path: Path) -> ClassifiedSourceDocument:
            pages = services.ocr_client.extract_pages(claim_id, pdf_path)
            if not pages:
                raise ValueError(f"Document {pdf_path.name} has no OCR pages")
            classification = services.classify_document(
                pdf_path.name,
                pages,
            )
            return ClassifiedSourceDocument(
                path=pdf_path,
                pages=pages,
                classification=classification,
            )

        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_WORKERS) as executor:
            classified_documents = list(executor.map(classify_source, pdf_paths))

    classified_documents.sort(
        key=lambda item: (
            item.classification.document_type.casefold(),
            item.path.name.casefold(),
            item.path.name,
        )
    )

    with log_step(log, "prepare_documents", root):
        logical_documents, pages, source_files = _prepare_folder_documents(
            claim_id,
            root,
            classified_documents,
        )
        _write_pages(root, pages)

    return _complete_ingestion(
        claim_id=claim_id,
        root=root,
        source_files=source_files,
        logical_documents=logical_documents,
        services=services,
        log=log,
        locked_document_types=True,
    )


def _collect_pdf_paths(folder_path: Path) -> list[Path]:
    if not folder_path.exists():
        raise FileNotFoundError(f"Document folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Document folder is not a directory: {folder_path}")
    pdf_paths = sorted(
        (
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not pdf_paths:
        raise ValueError(f"Document folder contains no PDF files: {folder_path}")
    return pdf_paths


def _prepare_folder_documents(
    claim_id: str,
    root: Path,
    source_documents: list[ClassifiedSourceDocument],
) -> tuple[list[LogicalDocument], list[PageText], list[Path]]:
    documents: list[LogicalDocument] = []
    all_pages: list[PageText] = []
    source_files: list[Path] = []
    next_page_number = 1

    for index, source in enumerate(source_documents, start=1):
        output_path = preserve_document_pdf(source.path, root)
        pages = []
        for page in sorted(source.pages, key=lambda item: item.page_number):
            pages.append(_renumber_page(claim_id, page, next_page_number))
            next_page_number += 1
        document_id = f"DOC-{index:03d}"
        documents.append(
            LogicalDocument(
                id=document_id,
                title=source.classification.title,
                document_type=source.classification.document_type,
                page_range=PageRange(
                    start_page=pages[0].page_number,
                    end_page=pages[-1].page_number,
                ),
                pages=pages,
                file_name=output_path.name,
            )
        )
        all_pages.extend(pages)
        source_files.append(output_path)

    return documents, all_pages, source_files


def _renumber_page(claim_id: str, page: PageText, page_number: int) -> PageText:
    return PageText(
        claim_id=claim_id,
        page_number=page_number,
        page_id=page_id_for(claim_id, page_number),
        text=page.text,
    )


def _write_pages(root: Path, pages: list[PageText]) -> None:
    write_jsonl(
        root / "pages.jsonl",
        [page.model_dump(mode="json") for page in pages],
    )


def _complete_ingestion(
    claim_id: str,
    root: Path,
    source_files: list[Path],
    logical_documents: list[LogicalDocument],
    services: IngestionServices,
    log: IngestionLog,
    locked_document_types: bool,
) -> ClaimManifest:
    retrieval_modes = {
        services.retrieval_mode,
        *services.additional_retrieval_modes,
    }
    with log_step(log, "chunk", root):
        chunks = chunk_documents(claim_id, logical_documents)

    with log_step(log, "metadata", root):
        documents, chunks = _extract_documents(
            logical_documents,
            chunks,
            services,
            locked_document_types,
        )

    embeddings: list[list[float]] = []
    if "semantic" in retrieval_modes:
        if services.embedder is None:
            raise ValueError("semantic retrieval requires an embedder")
        with log_step(log, "embed", root):
            embeddings = services.embedder.embed_texts(
                [chunk.text for chunk in chunks]
            )

    with log_step(log, "persist_chunks", root):
        write_jsonl(
            root / "chunks.jsonl",
            [chunk.model_dump(mode="json") for chunk in chunks],
        )

    if "semantic" in retrieval_modes:
        vector_store_factory = services.vector_store_factory
        if vector_store_factory is None:
            raise ValueError("semantic retrieval requires a vector store factory")
        with log_step(log, "index", root):
            _index_chunks(root, chunks, embeddings, vector_store_factory)
    if "lightrag" in retrieval_modes:
        if services.lightrag_indexer is None:
            raise ValueError("LightRAG retrieval requires an indexer")
        with log_step(log, "lightrag_index", root):
            services.lightrag_indexer(root / "index" / "lightrag", chunks)
    if retrieval_modes == {"lexical"}:
        with log_step(log, "clear_vector_index", root):
            _clear_vector_index(root)

    manifest = _build_manifest(
        claim_id,
        root,
        source_files,
        documents,
        len(chunks),
        services,
    )
    with log_step(log, "claim_manifest", root):
        write_claim_manifest(root, manifest)

    log.finished_at = log.entries[-1].finished_at
    write_json(root / "run_log.json", log.model_dump(mode="json"))
    return manifest


def _extract_documents(
    logical_documents: list[LogicalDocument],
    chunks: list[DocumentChunk],
    services: IngestionServices,
    locked_document_types: bool,
) -> tuple[list[DocumentMetadata], list[DocumentChunk]]:
    def extract_metadata(logical_document: LogicalDocument) -> DocumentMetadata:
        document_chunks = [
            chunk for chunk in chunks if chunk.document_id == logical_document.id
        ]
        metadata = services.extract_document_metadata(
            logical_document,
            document_chunks,
        )
        if not locked_document_types:
            return metadata
        if (
            metadata.document_type.casefold()
            != logical_document.document_type.casefold()
        ):
            raise ValueError(
                f"Document type changed after sorting for {logical_document.id}: "
                f"{logical_document.document_type!r} -> {metadata.document_type!r}"
            )
        return metadata.model_copy(
            update={"document_type": logical_document.document_type}
        )

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_WORKERS) as executor:
        documents = list(executor.map(extract_metadata, logical_documents))

    document_types = {
        document.id: document.document_type for document in documents
    }
    updated_chunks = [
        chunk.model_copy(
            update={"document_type": document_types[chunk.document_id]}
        )
        for chunk in chunks
    ]
    return documents, updated_chunks


def _index_chunks(
    root: Path,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    vector_store_factory: Callable[[Path], VectorStore],
) -> None:
    vector_store = vector_store_factory(root)
    try:
        vector_store.index_chunks(chunks, embeddings)
    finally:
        vector_store.close()


def _clear_vector_index(root: Path) -> None:
    index_path = root / "index"
    if index_path.exists():
        shutil.rmtree(index_path)


def _build_manifest(
    claim_id: str,
    root: Path,
    source_files: list[Path],
    documents: list[DocumentMetadata],
    chunk_count: int,
    services: IngestionServices,
) -> ClaimManifest:
    embedder = services.embedder
    uses_embeddings = any(
        mode in {"semantic", "lightrag"}
        for mode in (
            services.retrieval_mode,
            *services.additional_retrieval_modes,
        )
    )
    return ClaimManifest(
        claim_id=claim_id,
        source_files=[path.relative_to(root).as_posix() for path in source_files],
        documents=documents,
        chunk_count=chunk_count,
        embedding_provider=(
            services.embedding_provider
            or (embedder.embedding_provider if uses_embeddings and embedder else None)
        ),
        embedding_model=(
            services.embedding_model
            or (embedder.embedding_model if uses_embeddings and embedder else None)
        ),
        retrieval_mode=services.retrieval_mode,
        additional_retrieval_modes=list(services.additional_retrieval_modes),
    )


@contextmanager
def log_step(log: IngestionLog, step: str, root: Path) -> Iterator[None]:
    entry = IngestionLogEntry(step=step, status="running")
    log.entries.append(entry)
    LOGGER.info(
        "ingestion_step_start claim_id=%s step=%s root=%s",
        log.claim_id,
        step,
        root,
    )
    write_json(root / "run_log.json", log.model_dump(mode="json"))
    try:
        yield
    except Exception as exc:
        entry.status = "failed"
        entry.message = str(exc)
        entry.finished_at = utc_now()
        write_json(root / "run_log.json", log.model_dump(mode="json"))
        LOGGER.exception(
            "ingestion_step_failed claim_id=%s step=%s error=%s",
            log.claim_id,
            step,
            exc,
        )
        raise
    else:
        entry.status = "succeeded"
        entry.finished_at = utc_now()
        write_json(root / "run_log.json", log.model_dump(mode="json"))
        LOGGER.info(
            "ingestion_step_complete claim_id=%s step=%s",
            log.claim_id,
            step,
        )
