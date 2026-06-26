"""Claim knowledge base backend package."""

from claim_kb.api import (
    ClaimKbApi,
    ingest_claim_pdf,
    list_claim_documents,
    read_document_chunk,
    search_claim_file,
)
from claim_kb.schemas import (
    ChunkSearchResult,
    DocumentChunk,
    DocumentEvent,
    DocumentMetadata,
    DocumentParty,
    StructuredClaimFile,
)

__all__ = [
    "ChunkSearchResult",
    "ClaimKbApi",
    "DocumentChunk",
    "DocumentEvent",
    "DocumentMetadata",
    "DocumentParty",
    "StructuredClaimFile",
    "ingest_claim_pdf",
    "list_claim_documents",
    "read_document_chunk",
    "search_claim_file",
]
