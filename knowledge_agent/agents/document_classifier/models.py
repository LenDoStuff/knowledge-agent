"""Structured contracts for document classification."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from knowledge_agent.claims.models import (
    DocumentEvent,
    DocumentParty,
    PageRange,
    PageText,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Text = Annotated[str, StringConstraints(strip_whitespace=True)]


class PageBoundaryDecision(BaseModel):
    page_number: int = Field(ge=1)
    is_new_document: bool
    document_type: NonEmptyText
    title: NonEmptyText


class LogicalDocument(BaseModel):
    id: str
    title: str
    document_type: str
    page_range: PageRange
    pages: list[PageText]
    file_name: str | None = None


class ExtractedDocumentMetadata(BaseModel):
    title: NonEmptyText
    summary: Text = Field(description="A concise summary of no more than 200 words.")
    involved_parties: list[DocumentParty]
    events: list[DocumentEvent]
    document_type: NonEmptyText


class DocumentClassification(BaseModel):
    title: NonEmptyText
    document_type: NonEmptyText
