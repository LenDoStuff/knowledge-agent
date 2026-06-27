"""Claim PDF and document-folder ingestion orchestration."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from claim_kb.chunking import chunk_documents
from claim_kb.classify import ClaimClassifier, DocumentClassification
from claim_kb.config import EmbeddingMode, validate_embedding_mode
from claim_kb.embeddings import TextEmbedder
from claim_kb.filesystem import (
    ensure_claim_dirs,
    preserve_document_pdf,
    preserve_original_pdf,
    write_claim_metadata,
    write_json,
    write_jsonl,
)
from claim_kb.ocr import OcrClient
from claim_kb.schemas import (
    DocumentMetadata,
    IngestionLog,
    IngestionLogEntry,
    LogicalDocument,
    PageRange,
    PageText,
    StructuredClaimFile,
    page_id_for,
    utc_now,
)
from claim_kb.split import group_logical_documents, write_split_pdfs
from claim_kb.vector_store import VectorStore


@dataclass
class IngestionServices:
    ocr_client: OcrClient
    classifier: ClaimClassifier
    embedder: TextEmbedder | None
    vector_store_factory: Callable[[Path], VectorStore] | None
    embedding_mode: EmbeddingMode = "snowflake"
    close: Callable[[], None] = field(default=lambda: None, repr=False)


@dataclass(frozen=True)
class ClassifiedSourceDocument:
    path: Path
    pages: list[PageText]
    classification: DocumentClassification


def ingest_claim_pdf_with_services(
    claim_id: str,
    pdf_path: Path,
    data_root: Path,
    services: IngestionServices,
) -> StructuredClaimFile:
    try:
        return _ingest_claim_pdf(claim_id, pdf_path, data_root, services)
    finally:
        services.close()


def ingest_claim_folder_with_services(
    claim_id: str,
    folder_path: Path,
    data_root: Path,
    services: IngestionServices,
) -> StructuredClaimFile:
    try:
        return _ingest_claim_folder(claim_id, folder_path, data_root, services)
    finally:
        services.close()


def _ingest_claim_pdf(
    claim_id: str,
    pdf_path: Path,
    data_root: Path,
    services: IngestionServices,
) -> StructuredClaimFile:
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
            services.classifier,
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


def _ingest_claim_folder(
    claim_id: str,
    folder_path: Path,
    data_root: Path,
    services: IngestionServices,
) -> StructuredClaimFile:
    log = IngestionLog(claim_id=claim_id)
    root = ensure_claim_dirs(data_root, claim_id)

    with log_step(log, "collect_documents", root):
        pdf_paths = _collect_pdf_paths(folder_path)

    with log_step(log, "ocr_and_classify", root):
        classified_documents = []
        for pdf_path in pdf_paths:
            pages = services.ocr_client.extract_pages(claim_id, pdf_path)
            if not pages:
                raise ValueError(f"Document {pdf_path.name} has no OCR pages")
            classification = services.classifier.classify_document(
                pdf_path.name,
                pages,
            )
            classified_documents.append(
                ClassifiedSourceDocument(
                    path=pdf_path,
                    pages=pages,
                    classification=classification,
                )
            )

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
        width=page.width,
        height=page.height,
        unit=page.unit,
        word_count=page.word_count,
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
) -> StructuredClaimFile:
    embedding_mode = validate_embedding_mode(services.embedding_mode)
    with log_step(log, "chunk", root):
        chunks = chunk_documents(claim_id, logical_documents)

    with log_step(log, "metadata", root):
        documents: list[DocumentMetadata] = []
        for logical_document in logical_documents:
            document_chunks = [
                chunk
                for chunk in chunks
                if chunk.document_id == logical_document.id
            ]
            metadata = services.classifier.extract_document_metadata(
                logical_document,
                document_chunks,
            )
            if locked_document_types:
                if (
                    metadata.document_type.casefold()
                    != logical_document.document_type.casefold()
                ):
                    raise ValueError(
                        f"Document type changed after sorting for "
                        f"{logical_document.id}: "
                        f"{logical_document.document_type!r} -> "
                        f"{metadata.document_type!r}"
                    )
                metadata = metadata.model_copy(
                    update={"document_type": logical_document.document_type}
                )
            documents.append(metadata)
            for chunk in document_chunks:
                chunk.document_type = metadata.document_type

    if embedding_mode == "snowflake":
        if services.embedder is None:
            raise ValueError("Snowflake mode requires an embedder")
        with log_step(log, "embed", root):
            embeddings = services.embedder.embed_texts(
                [chunk.text for chunk in chunks]
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding

    with log_step(log, "persist_chunks", root):
        write_jsonl(
            root / "chunks.jsonl",
            [chunk.model_dump(mode="json") for chunk in chunks],
        )

    if embedding_mode == "snowflake":
        if services.vector_store_factory is None:
            raise ValueError("Snowflake mode requires a vector store factory")
        with log_step(log, "index", root):
            vector_store: VectorStore = services.vector_store_factory(root)
            try:
                vector_store.index_chunks(chunks)
            finally:
                vector_store.close()
    else:
        with log_step(log, "clear_vector_index", root):
            index_path = root / "index"
            if index_path.exists():
                shutil.rmtree(index_path)

    embedder = services.embedder
    is_snowflake = embedding_mode == "snowflake"
    claim_file = StructuredClaimFile(
        claim_id=claim_id,
        root_path=str(root),
        source_files=[str(path) for path in source_files],
        documents=documents,
        chunk_count=len(chunks),
        vector_store_path=(
            str(root / "index" / "chroma") if is_snowflake else None
        ),
        embedding_provider=(
            embedder.embedding_provider if is_snowflake and embedder else None
        ),
        embedding_model=(
            embedder.embedding_model if is_snowflake and embedder else None
        ),
        embedding_mode=embedding_mode,
    )
    with log_step(log, "claim_metadata", root):
        write_claim_metadata(root, claim_file)

    log.finished_at = log.entries[-1].finished_at
    write_json(root / "run_log.json", log.model_dump(mode="json"))
    return claim_file


@contextmanager
def log_step(log: IngestionLog, step: str, root: Path) -> Iterator[None]:
    entry = IngestionLogEntry(step=step, status="running")
    log.entries.append(entry)
    write_json(root / "run_log.json", log.model_dump(mode="json"))
    try:
        yield
    except Exception as exc:
        entry.status = "failed"
        entry.message = str(exc)
        entry.finished_at = utc_now()
        write_json(root / "run_log.json", log.model_dump(mode="json"))
        raise
    else:
        entry.status = "succeeded"
        entry.finished_at = utc_now()
        write_json(root / "run_log.json", log.model_dump(mode="json"))


def main() -> None:
    from claim_kb.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
