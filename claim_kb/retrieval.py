"""Claim document listing and vector retrieval functions."""

from __future__ import annotations

from claim_kb.api import ClaimKbApi
from claim_kb.schemas import ChunkSearchResult, DocumentChunk, DocumentMetadata


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
