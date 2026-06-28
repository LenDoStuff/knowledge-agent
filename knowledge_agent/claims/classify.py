"""LLM-assisted document classification and metadata extraction."""

from __future__ import annotations

import logging
from typing import Annotated, Protocol, TypeVar

from pydantic import BaseModel, Field, StringConstraints

from knowledge_agent.claims.models import (
    DocumentChunk,
    DocumentEvent,
    DocumentMetadata,
    DocumentParty,
    PageRange,
    PageText,
)
from knowledge_agent.llm.client import StructuredOutputClient


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Text = Annotated[str, StringConstraints(strip_whitespace=True)]
ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


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


class DocumentClassifier(Protocol):
    def classify_document(
        self,
        file_name: str,
        pages: list[PageText],
    ) -> "DocumentClassification":
        ...

    def classify_page_boundary(
        self,
        page: PageText,
        prior_page: PageText | None,
        current_document: LogicalDocument | None,
    ) -> PageBoundaryDecision:
        ...

    def extract_document_metadata(
        self,
        document: LogicalDocument,
        chunks: list[DocumentChunk],
    ) -> DocumentMetadata:
        ...


class ExtractedDocumentMetadata(BaseModel):
    title: NonEmptyText
    summary: Text
    involved_parties: list[DocumentParty]
    events: list[DocumentEvent]
    document_type: NonEmptyText


class DocumentClassification(BaseModel):
    title: NonEmptyText
    document_type: NonEmptyText


class ResponsesDocumentClassifier:
    def __init__(self, client: StructuredOutputClient) -> None:
        self._client = client

    def classify_document(
        self,
        file_name: str,
        pages: list[PageText],
    ) -> DocumentClassification:
        if not pages:
            raise ValueError(f"Document {file_name} has no OCR pages")
        document_text = "\n\n".join(
            f"Page {page.page_number}\n{page.text}" for page in pages
        )
        return self._parse(
            operation="classify_document",
            system=(
                "You classify complete documents in insurance claim files. "
                "Choose a concise title and a plain, specific document type."
            ),
            user=(
                "Classify this complete document.\n\n"
                f"File name: {file_name}\n\n"
                f"OCR text:\n{_clip(document_text, 10000)}"
            ),
            response_model=DocumentClassification,
        )

    def classify_page_boundary(
        self,
        page: PageText,
        prior_page: PageText | None,
        current_document: LogicalDocument | None,
    ) -> PageBoundaryDecision:
        if prior_page is None:
            return PageBoundaryDecision(
                page_number=page.page_number,
                is_new_document=True,
                document_type="unknown",
                title="Claim document",
            )

        current_context = "No current document."
        if current_document is not None:
            current_context = (
                f"Current document id: {current_document.id}\n"
                f"Current title: {current_document.title}\n"
                f"Current type: {current_document.document_type}\n"
                f"Current pages: {current_document.page_range.start_page}-"
                f"{current_document.page_range.end_page}"
            )
        decision = self._parse(
            operation="classify_page_boundary",
            system=(
                "You classify page boundaries in scanned insurance claim files. "
                "Decide whether the current page continues the current document "
                "or starts a new one."
            ),
            user=(
                "Classify the current page and use its number for page_number.\n\n"
                f"Prior page number: {prior_page.page_number}\n"
                f"Prior page text:\n{_clip(prior_page.text, 3000)}\n\n"
                f"Current page number: {page.page_number}\n"
                f"Current page text:\n{_clip(page.text, 3000)}\n\n"
                f"{current_context}"
            ),
            response_model=PageBoundaryDecision,
        )
        return decision.model_copy(update={"page_number": page.page_number})

    def extract_document_metadata(
        self,
        document: LogicalDocument,
        chunks: list[DocumentChunk],
    ) -> DocumentMetadata:
        if not chunks:
            raise ValueError(f"Document {document.id} has no chunks")
        chunk_text = "\n\n".join(
            f"Source ref: {chunk.source_ref}\n{chunk.text}" for chunk in chunks
        )
        extracted = self._parse(
            operation="extract_document_metadata",
            system=(
                "You extract concise metadata for logical documents in scanned "
                "insurance claim files."
            ),
            user=(
                "Extract a title, summary, involved parties, useful events, and "
                "document type. Every event must use a provided chunk source_ref. "
                "Keep the initial document type when it is not unknown.\n\n"
                f"Document id: {document.id}\n"
                f"Page range: {document.page_range.start_page}-"
                f"{document.page_range.end_page}\n"
                f"Initial title: {document.title}\n"
                f"Initial document_type: {document.document_type}\n\n"
                f"Chunks:\n{_clip(chunk_text, 10000)}"
            ),
            response_model=ExtractedDocumentMetadata,
        )
        valid_refs = {chunk.source_ref for chunk in chunks}
        for event in extracted.events:
            if event.source_ref not in valid_refs:
                raise ValueError(
                    f"Event source_ref is not a chunk in {document.id}: "
                    f"{event.source_ref}"
                )
        if document.file_name is None:
            raise ValueError(f"Document {document.id} has no PDF file name")
        return DocumentMetadata(
            id=document.id,
            title=extracted.title,
            summary=extracted.summary,
            involved_parties=extracted.involved_parties,
            events=extracted.events,
            document_type=extracted.document_type,
            page_range=document.page_range,
            file_name=document.file_name,
        )

    def _parse(
        self,
        operation: str,
        system: str,
        user: str,
        response_model: type[ParsedModel],
    ) -> ParsedModel:
        LOGGER.debug(
            "claim_classifier_prompt operation=%s response_model=%s "
            "system=%r user=%r",
            operation,
            response_model.__name__,
            system,
            user,
        )
        parsed = self._client.parse(system, user, response_model)
        LOGGER.debug(
            "claim_classifier_output operation=%s response_model=%s output=%s",
            operation,
            response_model.__name__,
            parsed.model_dump_json(),
        )
        return parsed


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"
