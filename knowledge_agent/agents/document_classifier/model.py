"""LLM-backed document classifier functions."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

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
from knowledge_agent.llm.providers import AgentRuntime


LOGGER = logging.getLogger(__name__)
AGENT_RETRIES = {"tools": 1, "output": 1}
ClassifierOutput = TypeVar("ClassifierOutput", bound=BaseModel)

DOCUMENT_CLASSIFIER_AGENT = Agent(
    name="document_classifier",
    output_type=DocumentClassification,
    retries=AGENT_RETRIES,
)
PAGE_BOUNDARY_AGENT = Agent(
    name="page_boundary_classifier",
    output_type=PageBoundaryDecision,
    retries=AGENT_RETRIES,
)
DOCUMENT_METADATA_AGENT = Agent(
    name="document_metadata_extractor",
    output_type=ExtractedDocumentMetadata,
    retries=AGENT_RETRIES,
)


def classify_document(
    runtime: AgentRuntime,
    file_name: str,
    pages: list[PageText],
) -> DocumentClassification:
    if not pages:
        raise ValueError(f"Document {file_name} has no OCR pages")
    system, user = build_classify_document_prompt(file_name, pages)
    return _run_classifier_agent(
        runtime,
        DOCUMENT_CLASSIFIER_AGENT,
        operation="classify_document",
        system=system,
        user=user,
    )


def classify_page_boundary(
    runtime: AgentRuntime,
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
    decision = _run_classifier_agent(
        runtime,
        PAGE_BOUNDARY_AGENT,
        operation="classify_page_boundary",
        system=system,
        user=user,
    )
    return decision.model_copy(update={"page_number": page.page_number})


def extract_document_metadata(
    runtime: AgentRuntime,
    document: LogicalDocument,
    chunks: list[DocumentChunk],
) -> DocumentMetadata:
    if not chunks:
        raise ValueError(f"Document {document.id} has no chunks")
    system, user = build_extract_metadata_prompt(document, chunks)
    extracted = _run_classifier_agent(
        runtime,
        DOCUMENT_METADATA_AGENT,
        operation="extract_document_metadata",
        system=system,
        user=user,
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


def _run_classifier_agent(
    runtime: AgentRuntime,
    agent: Agent[Any, ClassifierOutput],
    *,
    operation: str,
    system: str,
    user: str,
) -> ClassifierOutput:
    LOGGER.debug(
        "claim_classifier_prompt operation=%s system=%r user=%r",
        operation,
        system,
        user,
    )
    result = runtime.run(agent, user, instructions=system)
    parsed = result.output
    LOGGER.debug(
        "claim_classifier_output operation=%s output=%s usage=%s",
        operation,
        parsed.model_dump_json(),
        result.usage,
    )
    return parsed
