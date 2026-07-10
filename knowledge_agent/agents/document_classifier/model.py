"""LLM-backed document classifier functions."""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from knowledge_agent.agents.document_classifier.models import (
    DocumentClassification,
    ExtractedDocumentMetadata,
    LogicalDocument,
    PageBoundaryDecision,
)
from knowledge_agent.agents.document_classifier.prompts import (
    build_classify_document_prompt,
    build_extract_metadata_prompt,
    build_page_boundary_prompt,
)
from knowledge_agent.claims.models import DocumentChunk, DocumentMetadata, PageText
from knowledge_agent.llm.client import StructuredOutputParser


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


def classify_document(
    parse_structured_output: StructuredOutputParser,
    file_name: str,
    pages: list[PageText],
) -> DocumentClassification:
    if not pages:
        raise ValueError(f"Document {file_name} has no OCR pages")
    system, user = build_classify_document_prompt(file_name, pages)
    return _parse_classifier_output(
        parse_structured_output,
        operation="classify_document",
        system=system,
        user=user,
        response_model=DocumentClassification,
    )


def classify_page_boundary(
    parse_structured_output: StructuredOutputParser,
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

    system, user = build_page_boundary_prompt(
        page,
        prior_page,
        current_document,
    )
    decision = _parse_classifier_output(
        parse_structured_output,
        operation="classify_page_boundary",
        system=system,
        user=user,
        response_model=PageBoundaryDecision,
    )
    return decision.model_copy(update={"page_number": page.page_number})


def extract_document_metadata(
    parse_structured_output: StructuredOutputParser,
    document: LogicalDocument,
    chunks: list[DocumentChunk],
) -> DocumentMetadata:
    if not chunks:
        raise ValueError(f"Document {document.id} has no chunks")
    system, user = build_extract_metadata_prompt(document, chunks)
    extracted = _parse_classifier_output(
        parse_structured_output,
        operation="extract_document_metadata",
        system=system,
        user=user,
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


def _parse_classifier_output(
    parse_structured_output: StructuredOutputParser,
    *,
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
    parsed = parse_structured_output(system, user, response_model)
    LOGGER.debug(
        "claim_classifier_output operation=%s response_model=%s output=%s",
        operation,
        response_model.__name__,
        parsed.model_dump_json(),
    )
    return parsed
