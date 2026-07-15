"""Persisted claim access and lexical or semantic retrieval."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    RetrievalMode,
)
from knowledge_agent.claims.vector_store import VectorStore

if TYPE_CHECKING:
    from knowledge_agent.claims.lightrag import LightRagResource


@dataclass(frozen=True)
class ClaimStore:
    output_path: Path
    manifest: ClaimManifest
    retrieval_mode: RetrievalMode
    documents: list[DocumentMetadata]
    pages: list[PageText]
    chunks: list[DocumentChunk]
    documents_by_id: dict[str, DocumentMetadata]
    pages_by_id: dict[str, PageText]
    chunks_by_id: dict[str, DocumentChunk]
    embedder: TextEmbedder | None = None
    vector_store: VectorStore | None = None
    lightrag: LightRagResource | None = None


def load_claim_store(
    output_path: str | Path,
    *,
    retrieval_mode: RetrievalMode | None = None,
    validate_index: bool = True,
    embedder: TextEmbedder | None = None,
    vector_store: VectorStore | None = None,
    lightrag: LightRagResource | None = None,
) -> ClaimStore:
    output_path = Path(output_path)
    manifest_data = read_json(output_path / "manifest.json")
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest.json must contain a JSON object")
    manifest = ClaimManifest.model_validate(manifest_data)
    selected_mode = retrieval_mode or manifest.retrieval_mode
    if selected_mode not in manifest.available_retrieval_modes:
        raise ValueError(
            f"Retrieval mode {selected_mode!r} is not available for "
            f"claim {manifest.claim_id}"
        )
    if selected_mode == "lightrag" and validate_index:
        from knowledge_agent.claims.lightrag import validate_lightrag_index

        validate_lightrag_index(output_path / "index" / "lightrag", manifest)
    pages = [
        PageText.model_validate(row)
        for row in read_jsonl(output_path / "pages.jsonl")
    ]
    chunks = [
        DocumentChunk.model_validate(row)
        for row in read_jsonl(output_path / "chunks.jsonl")
    ]
    store = ClaimStore(
        output_path=output_path,
        manifest=manifest,
        retrieval_mode=selected_mode,
        documents=manifest.documents,
        pages=pages,
        chunks=chunks,
        documents_by_id={document.id: document for document in manifest.documents},
        pages_by_id={page.page_id: page for page in pages},
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
        embedder=embedder,
        vector_store=vector_store,
        lightrag=lightrag,
    )
    validate_claim_store(store)
    return store


def search_claim(
    store: ClaimStore,
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

    if store.retrieval_mode == "lightrag":
        raise RuntimeError(
            "LightRAG retrieval is asynchronous; use search_claim_evidence"
        )

    if store.retrieval_mode == "semantic":
        if store.embedder is None or store.vector_store is None:
            raise RuntimeError("semantic claim store requires retrieval dependencies")
        query_embedding = store.embedder.embed_texts([query])[0]
        hits = store.vector_store.search(
            query_embedding,
            document_types=document_types,
            top_k=top_k,
        )
        return [
            _search_result(store, get_chunk(store, hit.chunk_id), hit.score)
            for hit in hits
        ]

    allowed_types = set(document_types) if document_types else None
    ranked: list[tuple[int, int, DocumentChunk]] = []
    for index, chunk in enumerate(store.chunks):
        document = store.documents_by_id[chunk.document_id]
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
    return [
        _search_result(store, chunk, float(score))
        for score, _, chunk in ranked[:top_k]
    ]


def get_document(store: ClaimStore, document_id: str) -> DocumentMetadata:
    document = store.documents_by_id.get(document_id)
    if document is None:
        raise DocumentNotFoundError(f"Document not found: {document_id}")
    return document


def get_page(store: ClaimStore, page_id: str) -> PageText:
    page = store.pages_by_id.get(page_id)
    if page is None:
        raise PageNotFoundError(f"Page not found: {page_id}")
    return page


def get_chunk(store: ClaimStore, chunk_id: str) -> DocumentChunk:
    chunk = store.chunks_by_id.get(chunk_id)
    if chunk is None:
        raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
    return chunk


def validate_claim_store(store: ClaimStore) -> None:
    if len(store.documents_by_id) != len(store.documents):
        raise ValueError("manifest.json contains duplicate document IDs")
    if len(store.pages_by_id) != len(store.pages):
        raise ValueError("pages.jsonl contains duplicate page IDs")
    if len(store.chunks_by_id) != len(store.chunks):
        raise ValueError("chunks.jsonl contains duplicate chunk IDs")
    source_refs = {chunk.source_ref for chunk in store.chunks}
    if len(source_refs) != len(store.chunks):
        raise ValueError("chunks.jsonl contains duplicate source references")
    for page in store.pages:
        if page.claim_id != store.manifest.claim_id:
            raise ValueError(
                f"Page {page.page_id} belongs to {page.claim_id}, "
                f"expected {store.manifest.claim_id}"
            )
    for chunk in store.chunks:
        if chunk.document_id not in store.documents_by_id:
            raise ValueError(
                f"Chunk {chunk.chunk_id} references unknown document "
                f"{chunk.document_id}"
            )
        if chunk.claim_id != store.manifest.claim_id:
            raise ValueError(
                f"Chunk {chunk.chunk_id} belongs to {chunk.claim_id}, "
                f"expected {store.manifest.claim_id}"
            )
        missing_pages = [page for page in chunk.page_ids if page not in store.pages_by_id]
        if missing_pages:
            raise ValueError(
                f"Chunk {chunk.chunk_id} references unknown pages: {missing_pages}"
            )
    for document in store.documents:
        for event in document.events:
            if event.source_ref not in source_refs:
                raise ValueError(
                    f"Event in {document.id} references unknown chunk: "
                    f"{event.source_ref}"
                )


def _search_result(
    store: ClaimStore,
    chunk: DocumentChunk,
    score: float,
) -> ClaimSearchResult:
    document = store.documents_by_id[chunk.document_id]
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


def _terms(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())
