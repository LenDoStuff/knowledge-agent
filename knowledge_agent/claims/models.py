"""Persisted claim models and public retrieval results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RetrievalMode = Literal["lexical", "semantic", "lightrag"]
KnowledgeBaseEngine = Literal["custom", "lightrag"]


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

    @model_validator(mode="after")
    def validate_page_id(self) -> "PageText":
        expected = page_id_for(self.claim_id, self.page_number)
        if self.page_id != expected:
            raise ValueError(f"page_id must be {expected}")
        return self


class DocumentParty(BaseModel):
    name: str
    role: str

    @field_validator("name", "role", mode="before")
    @classmethod
    def require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("party fields must be non-empty strings")
        return value.strip()


class DocumentEvent(BaseModel):
    year: int | None = Field(default=None, ge=1)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)
    sentence: str
    source_ref: str

    @field_validator("sentence", "source_ref", mode="before")
    @classmethod
    def require_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("event fields must be non-empty strings")
        return value.strip()


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


class ClaimSearchResult(BaseModel):
    document_id: str
    chunk_id: str
    document_type: str
    document_title: str
    document_summary: str
    page_range: PageRange
    page_ids: list[str]
    source_ref: str
    text: str
    score: float


class ClaimManifest(BaseModel):
    claim_id: str
    source_files: list[str] = Field(min_length=1)
    documents: list[DocumentMetadata]
    chunk_count: int
    retrieval_mode: RetrievalMode
    embedding_provider: str | None = None
    embedding_model: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> "ClaimManifest":
        if self.retrieval_mode in {"semantic", "lightrag"} and (
            not self.embedding_provider or not self.embedding_model
        ):
            raise ValueError(
                f"{self.retrieval_mode} retrieval requires embedding_provider "
                "and embedding_model"
            )
        if self.retrieval_mode == "lexical" and (
            self.embedding_provider is not None or self.embedding_model is not None
        ):
            raise ValueError(
                "lexical retrieval cannot declare an embedding provider or model"
            )
        for document in self.documents:
            prefix = f"{self.claim_id}/{document.id}#"
            for event in document.events:
                if not event.source_ref.startswith(prefix):
                    raise ValueError(f"event source_ref must start with {prefix}")
        return self
