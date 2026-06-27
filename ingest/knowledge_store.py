"""Local lexical access to persisted Claim KB output."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ingest.exceptions import DocumentNotFoundError, PageNotFoundError
from ingest.schemas import (
    ChunkSearchResult,
    DocumentChunk,
    DocumentMetadata,
    KnowledgeItem,
    PageText,
    StructuredClaimFile,
)
from ingest.filesystem import read_json, read_jsonl


class ClaimKbKnowledgeStore:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        manifest_data = read_json(self.output_path / "manifest.json")
        if not isinstance(manifest_data, dict):
            raise ValueError("manifest.json must contain a JSON object")
        self.manifest = StructuredClaimFile.model_validate(manifest_data)

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
        if len(self._documents) != len(self.manifest.documents):
            raise ValueError("manifest.json contains duplicate document IDs")
        if len(self._pages) != len(pages):
            raise ValueError("pages.jsonl contains duplicate page IDs")
        for page in pages:
            if page.claim_id != self.manifest.claim_id:
                raise ValueError(
                    f"Page {page.page_id} has claim_id {page.claim_id}, "
                    f"expected {self.manifest.claim_id}"
                )

        chunk_refs = {chunk.source_ref for chunk in chunks}
        if len(chunk_refs) != len(chunks):
            raise ValueError("chunks.jsonl contains duplicate source references")

        self._items: list[KnowledgeItem] = []
        for chunk in chunks:
            document = self._documents.get(chunk.document_id)
            if document is None:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} references unknown document "
                    f"{chunk.document_id}"
                )
            if chunk.claim_id != self.manifest.claim_id:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has claim_id {chunk.claim_id}, "
                    f"expected {self.manifest.claim_id}"
                )
            missing_pages = [
                page_id
                for page_id in chunk.page_ids
                if page_id not in self._pages
            ]
            if missing_pages:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} references unknown pages: {missing_pages}"
                )
            self._items.append(
                KnowledgeItem(
                    item_id=chunk.chunk_id,
                    claim_id=chunk.claim_id,
                    document_id=chunk.document_id,
                    document_type=chunk.document_type,
                    document_title=document.title,
                    document_summary=document.summary,
                    text=chunk.text,
                    page_ids=chunk.page_ids,
                    source_ref=chunk.source_ref,
                )
            )

        for document in self.manifest.documents:
            for event in document.events:
                if event.source_ref not in chunk_refs:
                    raise ValueError(
                        f"Event in {document.id} references unknown chunk: "
                        f"{event.source_ref}"
                    )

    def search(self, query: str, top_k: int = 8) -> list[KnowledgeItem]:
        return [item for _, _, item in self._rank(query, None, top_k)]

    def search_chunks(
        self,
        query: str,
        document_types: list[str] | None = None,
        top_k: int = 8,
    ) -> list[ChunkSearchResult]:
        return [
            ChunkSearchResult(
                document_id=item.document_id,
                chunk_id=item.item_id,
                page_range=self._chunks[item.item_id].page_range,
                text=item.text,
                score=float(score),
                document_type=item.document_type,
            )
            for score, _, item in self._rank(query, document_types, top_k)
        ]

    def _rank(
        self,
        query: str,
        document_types: list[str] | None,
        top_k: int,
    ) -> list[tuple[int, int, KnowledgeItem]]:
        terms = _terms(query)
        if not terms:
            raise ValueError("query must contain searchable text")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        allowed_types = set(document_types) if document_types else None
        ranked: list[tuple[int, int, KnowledgeItem]] = []
        for index, item in enumerate(self._items):
            if allowed_types is not None and item.document_type not in allowed_types:
                continue
            searchable = " ".join(
                [
                    item.text,
                    item.document_title,
                    item.document_summary,
                    item.document_type,
                ]
            )
            counts = Counter(_terms(searchable))
            score = sum(counts[term] for term in terms)
            if score:
                ranked.append((score, index, item))

        ranked.sort(key=lambda result: (-result[0], result[1]))
        return ranked[:top_k]

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


def _terms(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())
