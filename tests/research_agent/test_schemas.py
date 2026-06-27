import pytest
from pydantic import ValidationError

from research_agent.schemas import (
    EvidenceItem,
    ResearchAnswer,
    ResearchFinding,
    ResearchQuery,
)


SOURCE_REF = "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"


def test_research_models_require_non_empty_text_and_sources():
    query = ResearchQuery(query="loss date", research_goal="Find the loss date")
    evidence = EvidenceItem(
        document_id="DOC-001",
        document_type="fnol",
        document_title="First Notice of Loss",
        page_ids=["CLM-SAMPLE-001:p1"],
        source_ref=SOURCE_REF,
        text="Loss date: 2026-06-01",
    )
    finding = ResearchFinding(
        insight="The loss date was June 1, 2026.",
        source_refs=[SOURCE_REF],
    )
    answer = ResearchAnswer(
        question="When was the loss?",
        answer=f"The loss was June 1, 2026. [{SOURCE_REF}]",
        findings=[finding],
        source_refs=[SOURCE_REF],
    )

    assert query.query == "loss date"
    assert evidence.source_ref == SOURCE_REF
    assert answer.source_refs == [SOURCE_REF]

    with pytest.raises(ValidationError):
        ResearchQuery(query=" ", research_goal="Find the loss date")
    with pytest.raises(ValidationError):
        ResearchQuery(query="loss date", research_goal=" ")
    with pytest.raises(ValidationError):
        EvidenceItem(
            document_id="DOC-001",
            document_type="fnol",
            document_title="First Notice of Loss",
            page_ids=[],
            source_ref=" ",
            text="",
        )
    with pytest.raises(ValidationError):
        ResearchFinding(insight=" ", source_refs=[SOURCE_REF])
    with pytest.raises(ValidationError):
        ResearchFinding(insight="A fact", source_refs=[])
    with pytest.raises(ValidationError):
        ResearchAnswer(
            question="When was the loss?",
            answer="The loss was June 1, 2026.",
            findings=[finding],
            source_refs=[],
        )
