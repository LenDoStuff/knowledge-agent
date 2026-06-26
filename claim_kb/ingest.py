"""Main orchestration and CLI for claim PDF ingestion."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from claim_kb.auth import create_browser_credential
from claim_kb.chunking import chunk_documents
from claim_kb.classify import AzureClaimClassifier, ClaimClassifier
from claim_kb.config import ClaimKbSettings
from claim_kb.embeddings import SnowflakeAiEmbedder, TextEmbedder
from claim_kb.ocr import AzureDocumentIntelligenceOcrClient, OcrClient
from claim_kb.schemas import (
    DocumentMetadata,
    IngestionLog,
    IngestionLogEntry,
    StructuredClaimFile,
    utc_now,
)
from claim_kb.split import group_logical_documents, write_split_pdfs
from claim_kb.storage import (
    ChromaVectorStore,
    VectorStore,
    ensure_claim_dirs,
    preserve_original_pdf,
    write_claim_metadata,
    write_document_inventory,
    write_json,
    write_jsonl,
)


def ingest_claim_pdf(claim_id: str, pdf_path: str) -> StructuredClaimFile:
    settings = ClaimKbSettings.from_env()
    settings.require_ingestion_settings()
    credential = create_browser_credential(settings)
    services = build_live_ingestion_services(claim_id, settings, credential)
    return ingest_claim_pdf_with_services(
        claim_id=claim_id,
        pdf_path=Path(pdf_path),
        settings=settings,
        services=services,
    )


def build_live_ingestion_services(
    claim_id: str,
    settings: ClaimKbSettings,
    credential: object,
) -> "IngestionServices":
    return IngestionServices(
        ocr_client=AzureDocumentIntelligenceOcrClient(settings, credential),
        classifier=AzureClaimClassifier(settings, credential),
        embedder=SnowflakeAiEmbedder(settings),
        vector_store_factory=lambda root: ChromaVectorStore(
            claim_id,
            root / "vector_store" / "chroma",
        ),
    )


class IngestionServices:
    def __init__(
        self,
        ocr_client: OcrClient,
        classifier: ClaimClassifier,
        embedder: TextEmbedder,
        vector_store_factory,
    ) -> None:
        self.ocr_client = ocr_client
        self.classifier = classifier
        self.embedder = embedder
        self.vector_store_factory = vector_store_factory


def ingest_claim_pdf_with_services(
    claim_id: str,
    pdf_path: Path,
    settings: ClaimKbSettings,
    services: IngestionServices,
) -> StructuredClaimFile:
    log = IngestionLog(claim_id=claim_id)
    root = ensure_claim_dirs(settings.data_root, claim_id)

    with log_step(log, "preserve_original", root):
        original_pdf = preserve_original_pdf(pdf_path, root)

    with log_step(log, "ocr", root):
        pages = services.ocr_client.extract_pages(claim_id, original_pdf)
        write_jsonl(
            root / "ocr" / "pages.jsonl",
            [page.model_dump(mode="json") for page in pages],
        )

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

    with log_step(log, "metadata", root):
        documents: list[DocumentMetadata] = []
        for logical_document in logical_documents:
            metadata = services.classifier.extract_document_metadata(logical_document)
            documents.append(metadata)
        write_document_inventory(root, documents)

    with log_step(log, "chunk", root):
        metadata_by_id = {document.id: document for document in documents}
        chunks = chunk_documents(claim_id, logical_documents, metadata_by_id)

    with log_step(log, "embed", root):
        try:
            embeddings = services.embedder.embed_texts([chunk.text for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding
            write_jsonl(
                root / "chunks" / "chunks.jsonl",
                [chunk.model_dump(mode="json") for chunk in chunks],
            )
        finally:
            close = getattr(services.embedder, "close", None)
            if close is not None:
                close()

    with log_step(log, "index", root):
        vector_store: VectorStore = services.vector_store_factory(root)
        try:
            vector_store.index_chunks(chunks)
        finally:
            close = getattr(vector_store, "close", None)
            if close is not None:
                close()

    claim_file = StructuredClaimFile(
        claim_id=claim_id,
        root_path=str(root),
        original_pdf_path=str(original_pdf),
        documents=documents,
        chunk_count=len(chunks),
        vector_store_path=str(root / "vector_store" / "chroma"),
        embedding_provider=getattr(services.embedder, "embedding_provider", "snowflake"),
        embedding_model=getattr(
            services.embedder,
            "embedding_model",
            settings.snowflake_embedding_model,
        ),
    )
    with log_step(log, "claim_metadata", root):
        write_claim_metadata(root, claim_file)

    log.finished_at = log.entries[-1].finished_at if log.entries else None
    write_json(root / "logs" / "ingestion_log.json", log.model_dump(mode="json"))
    return claim_file


@contextmanager
def log_step(log: IngestionLog, step: str, root: Path) -> Iterator[None]:
    entry = IngestionLogEntry(step=step, status="running")
    log.entries.append(entry)
    write_json(root / "logs" / "ingestion_log.json", log.model_dump(mode="json"))
    try:
        yield
    except Exception as exc:
        entry.status = "failed"
        entry.message = str(exc)
        entry.finished_at = utc_now()
        write_json(root / "logs" / "ingestion_log.json", log.model_dump(mode="json"))
        raise
    else:
        entry.status = "succeeded"
        entry.finished_at = utc_now()
        write_json(root / "logs" / "ingestion_log.json", log.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a scanned claim PDF.")
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--pdf-path", required=True)
    args = parser.parse_args()
    claim_file = ingest_claim_pdf(args.claim_id, args.pdf_path)
    print(claim_file.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
