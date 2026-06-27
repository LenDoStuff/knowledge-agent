"""Chroma vector-store protocol and adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from ingest.filesystem import safe_claim_id
from ingest.schemas import ChunkSearchResult, DocumentChunk, PageRange


class VectorStore(Protocol):
    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        ...

    def search(
        self,
        query_embedding: list[float],
        document_types: list[str] | None,
        top_k: int,
    ) -> list[ChunkSearchResult]:
        ...

    def close(self) -> None:
        ...


def chroma_collection_name(claim_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", safe_claim_id(claim_id))
    name = f"claim_{safe}"
    return name[:512]


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

    def reset_collection(self) -> None:
        client = self._client_instance()
        name = chroma_collection_name(self.claim_id)
        self._collection_cache = None
        existing_names = {collection.name for collection in client.list_collections()}
        if name in existing_names:
            client.delete_collection(name)

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.reset_collection()
        if not chunks:
            return
        collection = self._collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        document_types: list[str] | None,
        top_k: int,
    ) -> list[ChunkSearchResult]:
        collection = self._collection()
        where = None
        if document_types:
            where = {"document_type": {"$in": document_types}}
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        results: list[ChunkSearchResult] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index]
            distance = distances[index]
            score = 1.0 / (1.0 + float(distance))
            results.append(
                ChunkSearchResult(
                    document_id=str(metadata["document_id"]),
                    chunk_id=str(chunk_id),
                    page_range=PageRange(
                        start_page=int(metadata["page_start"]),
                        end_page=int(metadata["page_end"]),
                    ),
                    text=documents[index],
                    score=score,
                    document_type=str(metadata["document_type"]),
                )
            )
        return results

    def close(self) -> None:
        self._client = None
        self._collection_cache = None


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    return {
        "claim_id": chunk.claim_id,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "document_type": chunk.document_type,
        "page_start": chunk.page_range.start_page,
        "page_end": chunk.page_range.end_page,
        "chunk_index": chunk.chunk_index,
    }
