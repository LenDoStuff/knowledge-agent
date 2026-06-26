"""Pydantic models shared by ingestion, storage, and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PageRange(BaseModel):
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "PageRange":
        if self.end_page < self.start_page:
            raise ValueError("end_page must be greater than or equal to start_page")
        return self


class OcrPage(BaseModel):
    claim_id: str
    page_number: int = Field(ge=1)
    text: str = ""
    lines: list[str] = Field(default_factory=list)
    width: float | None = None
    height: float | None = None
    unit: str | None = None
    word_count: int = 0


class PageBoundaryDecision(BaseModel):
    page_number: int = Field(ge=1)
    is_new_document: bool
    document_type: str
    title: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("document_type", "title", "reason", mode="before")
    @classmethod
    def require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("text fields must be strings")
        text = value.strip()
        if not text:
            raise ValueError("text fields cannot be empty")
        return text


class LogicalDocument(BaseModel):
    id: str
    title: str
    document_type: str
    page_range: PageRange
    pages: list[OcrPage]
    file_name: str | None = None


class DocumentMetadata(BaseModel):
    id: str
    title: str
    summary: str
    involved_parties: list[str] = Field(default_factory=list)
    document_type: str
    page_range: PageRange
    file_name: str


class DocumentChunk(BaseModel):
    claim_id: str
    document_id: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    document_type: str
    page_range: PageRange
    text: str
    embedding: list[float] = Field(default_factory=list)


class ChunkSearchResult(BaseModel):
    document_id: str
    chunk_id: str
    page_range: PageRange
    text: str
    score: float
    document_type: str | None = None


class StructuredClaimFile(BaseModel):
    claim_id: str
    root_path: str
    original_pdf_path: str
    documents: list[DocumentMetadata]
    chunk_count: int
    vector_store_path: str
    embedding_provider: str
    embedding_model: str
    created_at: datetime = Field(default_factory=utc_now)


class IngestionLogEntry(BaseModel):
    step: str
    status: str
    message: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class IngestionLog(BaseModel):
    claim_id: str
    entries: list[IngestionLogEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
