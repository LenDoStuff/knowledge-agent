"""Stable programmatic API for claim knowledge base consumers.

Internal repo modules should prefer importing from this module instead of
depending on orchestration internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from claim_kb.auth import create_browser_credential
from claim_kb.config import ClaimKbSettings
from claim_kb.embeddings import SnowflakeAiEmbedder, TextEmbedder
from claim_kb.exceptions import ChunkNotFoundError, DocumentNotFoundError
from claim_kb.ingest import (
    IngestionServices,
    build_live_ingestion_services,
    ingest_claim_pdf_with_services,
)
from claim_kb.schemas import ChunkSearchResult, DocumentChunk, DocumentMetadata
from claim_kb.schemas import StructuredClaimFile
from claim_kb.storage import (
    ChromaVectorStore,
    VectorStore,
    claim_root,
    read_claim_metadata,
    read_chunks,
    read_document_inventory,
)


CredentialFactory = Callable[[ClaimKbSettings], object]
EmbedderFactory = Callable[[ClaimKbSettings, object | None], TextEmbedder]
VectorStoreFactory = Callable[[ClaimKbSettings, str], VectorStore]
IngestionServicesFactory = Callable[
    [str, ClaimKbSettings, object],
    IngestionServices,
]


class ClaimKbApi:
    """Programmatic facade for ingestion, listing, search, and chunk reads."""

    def __init__(
        self,
        settings: ClaimKbSettings | None = None,
        credential: object | None = None,
        credential_factory: CredentialFactory = create_browser_credential,
        ingestion_services_factory: IngestionServicesFactory = (
            build_live_ingestion_services
        ),
        embedder_factory: EmbedderFactory | None = None,
        vector_store_factory: VectorStoreFactory | None = None,
    ) -> None:
        self.settings = settings or ClaimKbSettings.from_env()
        self._credential = credential
        self._credential_factory = credential_factory
        self._ingestion_services_factory = ingestion_services_factory
        self._embedder_factory = embedder_factory or _build_live_embedder
        self._vector_store_factory = vector_store_factory or _build_live_vector_store

    def ingest_claim_pdf(
        self,
        claim_id: str,
        pdf_path: str | Path,
    ) -> StructuredClaimFile:
        self.settings.require_ingestion_settings()
        credential = self._get_credential()
        services = self._ingestion_services_factory(
            claim_id,
            self.settings,
            credential,
        )
        return ingest_claim_pdf_with_services(
            claim_id=claim_id,
            pdf_path=Path(pdf_path),
            settings=self.settings,
            services=services,
        )

    def list_claim_documents(self, claim_id: str) -> list[DocumentMetadata]:
        return read_document_inventory(self.settings.data_root, claim_id)

    def search_claim_file(
        self,
        claim_id: str,
        query: str,
        document_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[ChunkSearchResult]:
        self.settings.require_retrieval_settings()
        claim_file = read_claim_metadata(self.settings.data_root, claim_id)
        search_settings = replace(
            self.settings,
            snowflake_embedding_model=(
                claim_file.embedding_model or self.settings.snowflake_embedding_model
            ),
        )
        embedder = self._embedder_factory(search_settings, self._credential)
        vector_store = self._vector_store_factory(search_settings, claim_id)
        query_embedding = embedder.embed_texts([query])[0]
        try:
            return vector_store.search(query_embedding, document_types, top_k)
        finally:
            embedder_close = getattr(embedder, "close", None)
            if embedder_close is not None:
                embedder_close()
            close = getattr(vector_store, "close", None)
            if close is not None:
                close()

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

    def _get_credential(self) -> object:
        if self._credential is None:
            self._credential = self._credential_factory(self.settings)
        return self._credential


def ingest_claim_pdf(claim_id: str, pdf_path: str | Path) -> StructuredClaimFile:
    return ClaimKbApi().ingest_claim_pdf(claim_id, pdf_path)


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


def _build_live_embedder(
    settings: ClaimKbSettings,
    credential: object | None,
) -> TextEmbedder:
    return SnowflakeAiEmbedder(settings)


def _build_live_vector_store(
    settings: ClaimKbSettings,
    claim_id: str,
) -> VectorStore:
    root = claim_root(settings.data_root, claim_id)
    return ChromaVectorStore(claim_id, root / "vector_store" / "chroma")
