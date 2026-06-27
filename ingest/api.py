"""Stable programmatic API for claim knowledge base consumers.

Internal repo modules should prefer importing from this module instead of
depending on orchestration internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ingest.config import ClaimKbSettings
from ingest.embeddings import TextEmbedder
from ingest.exceptions import ChunkNotFoundError, DocumentNotFoundError
from ingest.filesystem import (
    claim_root,
    read_claim_metadata,
    read_chunks,
)
from ingest.knowledge_store import ClaimKbKnowledgeStore
from ingest.schemas import (
    ChunkSearchResult,
    DocumentChunk,
    DocumentMetadata,
    StructuredClaimFile,
)
from ingest.vector_store import VectorStore

if TYPE_CHECKING:
    from ingest.ingest import IngestionServices


EmbedderFactory = Callable[[ClaimKbSettings], TextEmbedder]
VectorStoreFactory = Callable[[ClaimKbSettings, str], VectorStore]
IngestionServicesFactory = Callable[
    [str, ClaimKbSettings],
    "IngestionServices",
]


class ClaimKbApi:
    """Programmatic facade for ingestion, listing, search, and chunk reads."""

    def __init__(
        self,
        settings: ClaimKbSettings | None = None,
        ingestion_services_factory: IngestionServicesFactory | None = None,
        embedder_factory: EmbedderFactory | None = None,
        vector_store_factory: VectorStoreFactory | None = None,
    ) -> None:
        from ingest.bootstrap import (
            build_live_embedder,
            build_live_ingestion_services,
            build_live_vector_store,
        )

        self.settings = settings if settings is not None else ClaimKbSettings.from_env()
        self._ingestion_services_factory = (
            ingestion_services_factory
            if ingestion_services_factory is not None
            else build_live_ingestion_services
        )
        self._embedder_factory = (
            embedder_factory if embedder_factory is not None else build_live_embedder
        )
        self._vector_store_factory = (
            vector_store_factory
            if vector_store_factory is not None
            else build_live_vector_store
        )

    def ingest_claim_pdf(
        self,
        claim_id: str,
        pdf_path: str | Path,
    ) -> StructuredClaimFile:
        from ingest.ingest import ingest_claim_pdf_with_services

        services = self._ingestion_services_factory(
            claim_id,
            self.settings,
        )
        return ingest_claim_pdf_with_services(
            claim_id=claim_id,
            pdf_path=Path(pdf_path),
            data_root=self.settings.data_root,
            services=services,
        )

    def ingest_claim_folder(
        self,
        claim_id: str,
        folder_path: str | Path,
    ) -> StructuredClaimFile:
        from ingest.ingest import ingest_claim_folder_with_services

        services = self._ingestion_services_factory(
            claim_id,
            self.settings,
        )
        return ingest_claim_folder_with_services(
            claim_id=claim_id,
            folder_path=Path(folder_path),
            data_root=self.settings.data_root,
            services=services,
        )

    def list_claim_documents(self, claim_id: str) -> list[DocumentMetadata]:
        return read_claim_metadata(self.settings.data_root, claim_id).documents

    def search_claim_file(
        self,
        claim_id: str,
        query: str,
        document_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[ChunkSearchResult]:
        claim_file = read_claim_metadata(self.settings.data_root, claim_id)
        if claim_file.embedding_mode == "none":
            store = ClaimKbKnowledgeStore(
                claim_root(self.settings.data_root, claim_id)
            )
            return store.search_chunks(
                query,
                document_types=document_types,
                top_k=top_k,
            )

        self.settings.require_retrieval_settings()
        if claim_file.embedding_model is None:
            raise ValueError("Snowflake claim manifest is missing embedding_model")
        search_settings = replace(
            self.settings,
            snowflake_embedding_model=claim_file.embedding_model,
        )
        embedder = self._embedder_factory(search_settings)
        try:
            vector_store = self._vector_store_factory(search_settings, claim_id)
            try:
                query_embedding = embedder.embed_texts([query])[0]
                return vector_store.search(query_embedding, document_types, top_k)
            finally:
                vector_store.close()
        finally:
            embedder.close()

    def read_document_chunk(
        self,
        claim_id: str,
        document_id: str,
        chunk_id: str,
    ) -> DocumentChunk:
        chunks = read_chunks(self.settings.data_root, claim_id)
        matching_document = [
            chunk for chunk in chunks if chunk.document_id == document_id
        ]
        if not matching_document:
            raise DocumentNotFoundError(
                f"Document {document_id} not found in claim {claim_id}"
            )
        for chunk in matching_document:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise ChunkNotFoundError(
            f"Chunk {chunk_id} not found in document {document_id} "
            f"for claim {claim_id}"
        )


def ingest_claim_pdf(
    claim_id: str,
    pdf_path: str | Path,
) -> StructuredClaimFile:
    return ClaimKbApi().ingest_claim_pdf(claim_id, pdf_path)


def ingest_claim_folder(
    claim_id: str,
    folder_path: str | Path,
) -> StructuredClaimFile:
    return ClaimKbApi().ingest_claim_folder(claim_id, folder_path)


def list_claim_documents(claim_id: str) -> list[DocumentMetadata]:
    return ClaimKbApi().list_claim_documents(claim_id)


def search_claim_file(
    claim_id: str,
    query: str,
    document_types: list[str] | None = None,
    top_k: int = 10,
) -> list[ChunkSearchResult]:
    return ClaimKbApi().search_claim_file(claim_id, query, document_types, top_k)


def read_document_chunk(
    claim_id: str,
    document_id: str,
    chunk_id: str,
) -> DocumentChunk:
    return ClaimKbApi().read_document_chunk(claim_id, document_id, chunk_id)
