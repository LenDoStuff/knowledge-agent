import asyncio
import logging
from contextlib import contextmanager
from typing import Any, cast

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from knowledge_agent.agents.document_classifier import (
    DocumentClassification,
    ExtractedDocumentMetadata,
    LogicalDocument,
    PageBoundaryDecision,
    classify_document,
    classify_page_boundary,
    extract_document_metadata,
)
from knowledge_agent.claims.models import DocumentChunk, PageRange, PageText
from knowledge_agent.llm.providers import AgentRuntime


@contextmanager
def output_runtime(output, observed):
    def model_function(messages, info: AgentInfo):
        observed.append((messages, info))
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, output.model_dump())]
        )

    runner = asyncio.Runner()
    try:
        yield AgentRuntime(
            model=FunctionModel(model_function),
            runner=runner,
            openai=cast(Any, None),
        )
    finally:
        runner.close()


def _prompt_text(messages) -> str:
    return "\n".join(
        str(getattr(part, "content", ""))
        for message in messages
        for part in message.parts
    )


def test_classify_complete_document_uses_pydantic_agent(caplog):
    observed = []
    output = DocumentClassification(
        title="Repair Invoice",
        document_type="invoice",
    )
    with output_runtime(output, observed) as runtime:
        with caplog.at_level(
            logging.DEBUG,
            logger="knowledge_agent.agents.document_classifier.model",
        ):
            result = classify_document(
                runtime,
                "repair_invoice.pdf",
                [
                    PageText(
                        claim_id="CLM-001",
                        page_number=1,
                        page_id="CLM-001:p1",
                        text="Repair Invoice\nTotal: 850.00",
                    )
                ],
            )

    messages, info = observed[0]
    prompt = _prompt_text(messages)
    assert info.output_tools[0].name == "final_result"
    assert "repair_invoice.pdf" in prompt
    assert "Repair Invoice" in prompt
    assert result.document_type == "invoice"
    assert "claim_classifier_prompt operation=classify_document" in caplog.text
    assert "claim_classifier_output operation=classify_document" in caplog.text


def test_classify_document_rejects_empty_pages():
    with pytest.raises(ValueError, match="has no OCR pages"):
        classify_document(cast(AgentRuntime, None), "empty.pdf", [])


def test_classify_page_boundary_uses_structured_agent_output():
    observed = []
    output = PageBoundaryDecision(
        page_number=999,
        is_new_document=True,
        document_type="invoice",
        title="Repair Invoice",
    )
    with output_runtime(output, observed) as runtime:
        decision = classify_page_boundary(
            runtime,
            page=PageText(
                claim_id="CLM-001",
                page_number=2,
                page_id="CLM-001:p2",
                text="Repair Invoice\nTotal: 850.00",
            ),
            prior_page=PageText(
                claim_id="CLM-001",
                page_number=1,
                page_id="CLM-001:p1",
                text="First Notice of Loss",
            ),
            current_document=LogicalDocument(
                id="DOC-001",
                title="First Notice of Loss",
                document_type="fnol",
                page_range=PageRange(start_page=1, end_page=1),
                pages=[],
            ),
        )

    assert decision.page_number == 2
    assert decision.document_type == "invoice"
    assert "Repair Invoice" in _prompt_text(observed[0][0])


def test_extract_document_metadata_validates_exact_event_sources():
    source_ref = "CLM-001/DOC-002#DOC-002-CHUNK-001"
    output = ExtractedDocumentMetadata(
        title="Repair Invoice",
        summary="Invoice for sample repair work.",
        involved_parties=[{"name": "Sample Body Shop", "role": "repair vendor"}],
        events=[
            {
                "year": 2026,
                "month": 6,
                "day": None,
                "sentence": "Sample Body Shop listed repair work.",
                "source_ref": source_ref,
            }
        ],
        document_type="invoice",
    )
    document = LogicalDocument(
        id="DOC-002",
        title="Invoice",
        document_type="unknown",
        page_range=PageRange(start_page=2, end_page=2),
        pages=[],
        file_name="DOC-002_invoice.pdf",
    )
    chunk = DocumentChunk(
        claim_id="CLM-001",
        document_id="DOC-002",
        chunk_id="DOC-002-CHUNK-001",
        source_ref=source_ref,
        chunk_index=0,
        document_type="unknown",
        page_range=PageRange(start_page=2, end_page=2),
        page_ids=["CLM-001:p2"],
        text="Repair Invoice",
    )

    with output_runtime(output, []) as runtime:
        metadata = extract_document_metadata(runtime, document, [chunk])
    assert metadata.id == "DOC-002"
    assert metadata.events[0].source_ref == source_ref

    invalid = output.model_copy(deep=True)
    invalid.events[0].source_ref += ":p2"
    with output_runtime(invalid, []) as runtime:
        with pytest.raises(ValueError, match="not a chunk"):
            extract_document_metadata(runtime, document, [chunk])
