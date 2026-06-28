"""Chroma vector index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from knowledge_agent.claims.filesystem import safe_claim_id
from knowledge_agent.claims.models import DocumentChunk


@dataclass(frozen=True)
class VectorSearchHit:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        ...

    def search(
        self,
        query_embedding: list[float],
        document_types: list[str] | None,
        top_k: int,
    ) -> list[VectorSearchHit]:
        ...

    def close(self) -> None:
        ...


def chroma_collection_name(claim_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", safe_claim_id(claim_id))
    return f"claim_{safe}"[:512]


class ChromaVectorStore:
    def __init__(self, claim_id: str, persist_path: Path) -> None:
        self.claim_id = claim_id
        self.persist_path = persist_path
        self._client = None
        self._collection_cache = None

    def _collection(self):
        if self._collection_cache is not None:
            return self._collection_cache
        client = self._client_instance()
        self._collection_cache = client.get_or_create_collection(
            name=chroma_collection_name(self.claim_id),
            metadata={"claim_id": self.claim_id},
        )
        return self._collection_cache

    def _client_instance(self):
        if self._client is not None:
            return self._client
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        return self._client

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("each chunk requires exactly one embedding")
        client = self._client_instance()
        name = chroma_collection_name(self.claim_id)
        self._collection_cache = None
        if name in {collection.name for collection in client.list_collections()}:
            client.delete_collection(name)
        if not chunks:
            return
        collection = self._collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        document_types: list[str] | None,
        top_k: int,
    ) -> list[VectorSearchHit]:
        where = None
        if document_types:
            where = {"document_type": {"$in": document_types}}
        response = self._collection().query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["distances"],
        )
        return [
            VectorSearchHit(chunk_id=str(chunk_id), score=1.0 / (1.0 + float(distance)))
            for chunk_id, distance in zip(
                response["ids"][0],
                response["distances"][0],
                strict=True,
            )
        ]

    def close(self) -> None:
        self._client = None
        self._collection_cache = None


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    return {
        "claim_id": chunk.claim_id,
        "document_id": chunk.document_id,
        "document_type": chunk.document_type,
        "page_start": chunk.page_range.start_page,
        "page_end": chunk.page_range.end_page,
        "chunk_index": chunk.chunk_index,
    }
