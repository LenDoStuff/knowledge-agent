"""Persisted claim access and lexical or semantic retrieval."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from knowledge_agent.claims.embeddings import TextEmbedder
from knowledge_agent.claims.errors import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    PageNotFoundError,
)
from knowledge_agent.claims.filesystem import read_json, read_jsonl
from knowledge_agent.claims.models import (
    ClaimManifest,
    ClaimSearchResult,
    DocumentChunk,
    DocumentMetadata,
    PageText,
)
from knowledge_agent.claims.vector_store import VectorStore


class ClaimStore:
    def __init__(
        self,
        output_path: str | Path,
        *,
        embedder: TextEmbedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        manifest_data = read_json(self.output_path / "manifest.json")
        if not isinstance(manifest_data, dict):
            raise ValueError("manifest.json must contain a JSON object")
        self.manifest = ClaimManifest.model_validate(manifest_data)
        pages = [
            PageText.model_validate(row)
            for row in read_jsonl(self.output_path / "pages.jsonl")
        ]
        chunks = [
            DocumentChunk.model_validate(row)
            for row in read_jsonl(self.output_path / "chunks.jsonl")
        ]

        self._documents = {document.id: document for document in self.manifest.documents}
        self._pages = {page.page_id: page for page in pages}
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunk_order = [chunk.chunk_id for chunk in chunks]
        self._embedder = embedder
        self._vector_store = vector_store
        self._validate(pages, chunks)

    @property
    def documents(self) -> list[DocumentMetadata]:
        return self.manifest.documents

    @property
    def pages(self) -> list[PageText]:
        return list(self._pages.values())

    @property
    def chunks(self) -> list[DocumentChunk]:
        return [self._chunks[chunk_id] for chunk_id in self._chunk_order]

    def search(
        self,
        query: str,
        *,
        document_types: list[str] | None = None,
        top_k: int = 8,
    ) -> list[ClaimSearchResult]:
        terms = _terms(query)
        if not terms:
            raise ValueError("query must contain searchable text")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self.manifest.retrieval_mode == "semantic":
            if self._embedder is None or self._vector_store is None:
                raise RuntimeError("semantic claim store requires retrieval dependencies")
            query_embedding = self._embedder.embed_texts([query])[0]
            hits = self._vector_store.search(
                query_embedding,
                document_types=document_types,
                top_k=top_k,
            )
            return [self._result(self.get_chunk(hit.chunk_id), hit.score) for hit in hits]

        allowed_types = set(document_types) if document_types else None
        ranked: list[tuple[int, int, DocumentChunk]] = []
        for index, chunk_id in enumerate(self._chunk_order):
            chunk = self._chunks[chunk_id]
            document = self._documents[chunk.document_id]
            if allowed_types is not None and chunk.document_type not in allowed_types:
                continue
            searchable = " ".join(
                [chunk.text, document.title, document.summary, chunk.document_type]
            )
            counts = Counter(_terms(searchable))
            score = sum(counts[term] for term in terms)
            if score:
                ranked.append((score, index, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [self._result(chunk, float(score)) for score, _, chunk in ranked[:top_k]]

    def get_document(self, document_id: str) -> DocumentMetadata:
        document = self._documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return document

    def get_page(self, page_id: str) -> PageText:
        page = self._pages.get(page_id)
        if page is None:
            raise PageNotFoundError(f"Page not found: {page_id}")
        return page

    def get_chunk(self, chunk_id: str) -> DocumentChunk:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
        return chunk

    def _result(self, chunk: DocumentChunk, score: float) -> ClaimSearchResult:
        document = self._documents[chunk.document_id]
        return ClaimSearchResult(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            document_type=chunk.document_type,
            document_title=document.title,
            document_summary=document.summary,
            page_range=chunk.page_range,
            page_ids=chunk.page_ids,
            source_ref=chunk.source_ref,
            text=chunk.text,
            score=score,
        )

    def _validate(
        self,
        pages: list[PageText],
        chunks: list[DocumentChunk],
    ) -> None:
        if len(self._documents) != len(self.manifest.documents):
            raise ValueError("manifest.json contains duplicate document IDs")
        if len(self._pages) != len(pages):
            raise ValueError("pages.jsonl contains duplicate page IDs")
        if len(self._chunks) != len(chunks):
            raise ValueError("chunks.jsonl contains duplicate chunk IDs")
        source_refs = {chunk.source_ref for chunk in chunks}
        if len(source_refs) != len(chunks):
            raise ValueError("chunks.jsonl contains duplicate source references")
        for page in pages:
            if page.claim_id != self.manifest.claim_id:
                raise ValueError(
                    f"Page {page.page_id} belongs to {page.claim_id}, "
                    f"expected {self.manifest.claim_id}"
                )
        for chunk in chunks:
            if chunk.document_id not in self._documents:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} references unknown document "
                    f"{chunk.document_id}"
                )
            if chunk.claim_id != self.manifest.claim_id:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} belongs to {chunk.claim_id}, "
                    f"expected {self.manifest.claim_id}"
                )
            missing_pages = [page for page in chunk.page_ids if page not in self._pages]
            if missing_pages:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} references unknown pages: {missing_pages}"
                )
        for document in self.manifest.documents:
            for event in document.events:
                if event.source_ref not in source_refs:
                    raise ValueError(
                        f"Event in {document.id} references unknown chunk: "
                        f"{event.source_ref}"
                    )


def _terms(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())
