from types import SimpleNamespace

import pytest

from claim_kb.classify import AzureClaimClassifier, ExtractedDocumentMetadata
from claim_kb.schemas import (
    DocumentChunk,
    LogicalDocument,
    PageBoundaryDecision,
    PageRange,
    PageText,
)


class FakeResponses:
    def __init__(self, parsed_outputs):
        self.parsed_outputs = list(parsed_outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.parsed_outputs.pop(0)
        return SimpleNamespace(output_parsed=parsed)

    def create(self, **kwargs):
        raise AssertionError("classifier should use responses.parse, not create")


def build_classifier(parsed_outputs):
    responses = FakeResponses(parsed_outputs)
    classifier = AzureClaimClassifier.__new__(AzureClaimClassifier)
    classifier._model = "gpt-test"
    classifier._client = SimpleNamespace(
        responses=responses,
    )
    return classifier, responses


def test_classify_page_boundary_uses_responses_structured_parse():
    classifier, responses = build_classifier(
        [
            PageBoundaryDecision(
                page_number=999,
                is_new_document=True,
                document_type="invoice",
                title="Repair Invoice",
                reason="Invoice heading starts a new document.",
                confidence=0.93,
            )
        ]
    )

    decision = classifier.classify_page_boundary(
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

    call = responses.calls[0]
    prompt_text = "\n".join(message["content"] for message in call["input"])
    assert call["text_format"] is PageBoundaryDecision
    assert "json" not in prompt_text.lower()
    assert call["model"] == "gpt-test"
    assert decision.page_number == 2
    assert decision.document_type == "invoice"


def test_extract_document_metadata_uses_responses_structured_parse():
    classifier, responses = build_classifier(
        [
            ExtractedDocumentMetadata(
                title="Repair Invoice",
                summary="Invoice for sample repair work.",
                involved_parties=[
                    {"name": "Sample Body Shop", "role": "repair vendor"},
                ],
                events=[
                    {
                        "year": 2026,
                        "month": 6,
                        "day": None,
                        "sentence": "Sample Body Shop listed repair work.",
                        "source_ref": "CLM-001/DOC-002#DOC-002-CHUNK-001",
                    },
                ],
                document_type="invoice",
            )
        ]
    )
    document = LogicalDocument(
        id="DOC-002",
        title="Invoice",
        document_type="unknown",
        page_range=PageRange(start_page=2, end_page=2),
        pages=[
            PageText(
                claim_id="CLM-001",
                page_number=2,
                page_id="CLM-001:p2",
                text="Repair Invoice\nVendor: Sample Body Shop",
            )
        ],
        file_name="DOC-002_invoice.pdf",
    )

    chunks = [
        DocumentChunk(
            claim_id="CLM-001",
            document_id="DOC-002",
            chunk_id="DOC-002-CHUNK-001",
            source_ref="CLM-001/DOC-002#DOC-002-CHUNK-001",
            chunk_index=0,
            document_type="unknown",
            page_range=PageRange(start_page=2, end_page=2),
            page_ids=["CLM-001:p2"],
            text="Repair Invoice\nVendor: Sample Body Shop",
        )
    ]

    metadata = classifier.extract_document_metadata(document, chunks)

    call = responses.calls[0]
    prompt_text = "\n".join(message["content"] for message in call["input"])
    assert call["text_format"] is ExtractedDocumentMetadata
    assert "json" not in prompt_text.lower()
    assert call["model"] == "gpt-test"
    assert metadata.id == "DOC-002"
    assert metadata.page_range == PageRange(start_page=2, end_page=2)
    assert metadata.file_name == "DOC-002_invoice.pdf"
    assert metadata.involved_parties[0].role == "repair vendor"
    assert metadata.events[0].year == 2026
    assert metadata.events[0].month == 6
    assert metadata.events[0].day is None
    assert metadata.events[0].source_ref == chunks[0].source_ref


def test_parse_response_raises_when_parsed_output_is_missing():
    classifier, _ = build_classifier([None])

    with pytest.raises(ValueError, match="Expected parsed structured output"):
        classifier._parse_response("system", "user", ExtractedDocumentMetadata)
