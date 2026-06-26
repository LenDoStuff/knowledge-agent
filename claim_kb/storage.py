"""Filesystem and Chroma persistence helpers."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Protocol

from claim_kb.exceptions import ClaimNotFoundError
from claim_kb.schemas import (
    ChunkSearchResult,
    DocumentChunk,
    DocumentMetadata,
    PageRange,
    StructuredClaimFile,
)


CLAIM_SUBDIRS = [
    "original",
    "documents",
    "metadata",
    "ocr",
    "chunks",
    "vector_store/chroma",
    "logs",
]


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


def safe_claim_id(claim_id: str) -> str:
    claim_id = claim_id.strip()
    if not claim_id:
        raise ValueError("claim_id cannot be empty")
    if any(sep in claim_id for sep in ("/", "\\")) or claim_id in {".", ".."}:
        raise ValueError("claim_id cannot contain path separators")
    return claim_id


def claim_root(data_root: Path, claim_id: str) -> Path:
    return data_root / safe_claim_id(claim_id)


def ensure_claim_dirs(data_root: Path, claim_id: str) -> Path:
    root = claim_root(data_root, claim_id)
    for subdir in CLAIM_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def require_claim_root(data_root: Path, claim_id: str) -> Path:
    root = claim_root(data_root, claim_id)
    if not root.exists():
        raise ClaimNotFoundError(f"Claim not found: {claim_id}")
    return root


def preserve_original_pdf(pdf_path: Path, root: Path) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    destination = root / "original" / "original_claim_file.pdf"
    shutil.copy2(pdf_path, destination)
    return destination


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, items: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_claim_metadata(root: Path, claim_file: StructuredClaimFile) -> None:
    write_json(root / "metadata" / "claim_file.json", claim_file.model_dump(mode="json"))


def read_claim_metadata(data_root: Path, claim_id: str) -> StructuredClaimFile:
    root = require_claim_root(data_root, claim_id)
    path = root / "metadata" / "claim_file.json"
    if not path.exists():
        raise ClaimNotFoundError(f"Claim metadata not found for claim: {claim_id}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid claim metadata: {path}")
    return StructuredClaimFile.model_validate(data)


def write_document_inventory(root: Path, documents: list[DocumentMetadata]) -> None:
    write_json(
        root / "metadata" / "document_inventory.json",
        [document.model_dump(mode="json") for document in documents],
    )


def read_document_inventory(data_root: Path, claim_id: str) -> list[DocumentMetadata]:
    root = require_claim_root(data_root, claim_id)
    path = root / "metadata" / "document_inventory.json"
    if not path.exists():
        raise ClaimNotFoundError(f"Document inventory not found for claim: {claim_id}")
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid document inventory: {path}")
    return [DocumentMetadata.model_validate(item) for item in data]


def read_chunks(data_root: Path, claim_id: str) -> list[DocumentChunk]:
    root = require_claim_root(data_root, claim_id)
    return [
        DocumentChunk.model_validate(row)
        for row in read_jsonl(root / "chunks" / "chunks.jsonl")
    ]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Cannot create slug from empty value")
    return slug


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
