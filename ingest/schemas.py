"""Pydantic models shared by ingestion, storage, and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ingest.config import EmbeddingMode


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


def page_id_for(claim_id: str, page_number: int) -> str:
    return f"{claim_id}:p{page_number}"


def source_ref_for(claim_id: str, document_id: str, chunk_id: str) -> str:
    return f"{claim_id}/{document_id}#{chunk_id}"


class PageText(BaseModel):
    claim_id: str
    page_number: int = Field(ge=1)
    page_id: str
    text: str = ""
    width: float | None = None
    height: float | None = None
    unit: str | None = None
    word_count: int = 0

    @model_validator(mode="after")
    def validate_page_id(self) -> "PageText":
        expected = page_id_for(self.claim_id, self.page_number)
        if self.page_id != expected:
            raise ValueError(f"page_id must be {expected}")
        return self


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
    pages: list[PageText]
    file_name: str | None = None


class DocumentParty(BaseModel):
    name: str
    role: str

    @field_validator("name", "role", mode="before")
    @classmethod
    def require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("party fields must be strings")
        text = value.strip()
        if not text:
            raise ValueError("party fields cannot be empty")
        return text


class DocumentEvent(BaseModel):
    year: int | None = Field(default=None, ge=1)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)
    sentence: str
    source_ref: str

    @field_validator("sentence", "source_ref", mode="before")
    @classmethod
    def require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("event text fields must be strings")
        text = value.strip()
        if not text:
            raise ValueError("event text fields cannot be empty")
        return text


class DocumentMetadata(BaseModel):
    id: str
    title: str
    summary: str
    involved_parties: list[DocumentParty] = Field(default_factory=list)
    events: list[DocumentEvent] = Field(default_factory=list)
    document_type: str
    page_range: PageRange
    file_name: str


class DocumentChunk(BaseModel):
    claim_id: str
    document_id: str
    chunk_id: str
    source_ref: str
    chunk_index: int = Field(ge=0)
    document_type: str
    page_range: PageRange
    page_ids: list[str]
    text: str
    embedding: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_citation_fields(self) -> "DocumentChunk":
        expected_ref = source_ref_for(
            self.claim_id,
            self.document_id,
            self.chunk_id,
        )
        if self.source_ref != expected_ref:
            raise ValueError(f"source_ref must be {expected_ref}")
        if not self.page_ids:
            raise ValueError("page_ids cannot be empty")
        prefix = f"{self.claim_id}:p"
        try:
            page_numbers = [
                int(page_id.removeprefix(prefix))
                for page_id in self.page_ids
                if page_id.startswith(prefix)
            ]
        except ValueError as exc:
            raise ValueError("page_ids must end with a page number") from exc
        if len(page_numbers) != len(self.page_ids):
            raise ValueError(f"page_ids must start with {prefix}")
        if page_numbers != sorted(set(page_numbers)):
            raise ValueError("page_ids must be unique and ordered")
        if (
            page_numbers[0] != self.page_range.start_page
            or page_numbers[-1] != self.page_range.end_page
        ):
            raise ValueError("page_ids must match the chunk page_range")
        return self


class KnowledgeItem(BaseModel):
    item_id: str
    claim_id: str
    document_id: str
    document_type: str
    document_title: str
    document_summary: str
    text: str
    page_ids: list[str]
    source_ref: str


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
    source_files: list[str] = Field(min_length=1)
    documents: list[DocumentMetadata]
    chunk_count: int
    vector_store_path: str | None
    embedding_provider: str | None
    embedding_model: str | None
    embedding_mode: EmbeddingMode = "snowflake"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_event_document_refs(self) -> "StructuredClaimFile":
        for document in self.documents:
            prefix = f"{self.claim_id}/{document.id}#"
            for event in document.events:
                if not event.source_ref.startswith(prefix):
                    raise ValueError(
                        f"event source_ref must start with {prefix}"
                    )
        return self


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
