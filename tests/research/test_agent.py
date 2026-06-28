import logging
from pathlib import Path

import pytest

from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.research.agent import run_claim_research
from knowledge_agent.research.models import ResearchFinding, ResearchQuery


SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
)
INVOICE_REF = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
FNOL_REF = "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"


class FakeResearchModel:
    def __init__(self) -> None:
        self.plan_calls = []
        self.extraction_calls = []
        self.answer_findings = []

    def plan_queries(self, question, documents, queries_per_question):
        self.plan_calls.append((question, documents, queries_per_question))
        if question == "When did the collision occur?":
            return [
                ResearchQuery(
                    query="loss date collision",
                    research_goal="Find the collision date.",
                )
            ]
        return [
            ResearchQuery(
                query="repair invoice",
                research_goal="Identify invoiced repair work.",
            )
        ]

    def extract_findings(self, query, evidence):
        self.extraction_calls.append((query, evidence))
        if query.query == "loss date collision":
            return [
                ResearchFinding(
                    insight="The collision occurred on June 1, 2026.",
                    source_refs=[FNOL_REF],
                )
            ]
        finding = ResearchFinding(
            insight="The invoice lists labor and a front bumper cover.",
            source_refs=[INVOICE_REF],
            follow_up_questions=["When did the collision occur?"],
        )
        return [finding, finding.model_copy()]

    def write_answer(self, question, findings):
        self.answer_findings = findings
        return f"The invoice lists labor and a front bumper cover. [{INVOICE_REF}]"


def test_run_claim_research_uses_manifest_and_retrieved_evidence():
    model = FakeResearchModel()

    answer = run_claim_research(
        store=ClaimStore(SAMPLE_OUTPUT),
        question="What repairs were invoiced?",
        model=model,
        max_depth=1,
    )

    planned_documents = model.plan_calls[0][1]
    evidence = model.extraction_calls[0][1]
    assert [document.id for document in planned_documents] == ["DOC-001", "DOC-002"]
    assert evidence[0].source_ref == INVOICE_REF
    assert "front bumper cover" in evidence[0].text
    assert answer.source_refs == [INVOICE_REF]
    assert answer.findings == model.answer_findings
    assert len(answer.findings) == 1


def test_second_layer_reduces_queries_per_question():
    model = FakeResearchModel()

    answer = run_claim_research(
        store=ClaimStore(SAMPLE_OUTPUT),
        question="What repairs were invoiced?",
        model=model,
        queries_per_question=4,
        max_depth=2,
    )

    assert [call[0] for call in model.plan_calls] == [
        "What repairs were invoiced?",
        "When did the collision occur?",
    ]
    assert [call[2] for call in model.plan_calls] == [4, 2]
    assert len(answer.findings) == 2
    assert answer.source_refs == [INVOICE_REF, FNOL_REF]


def test_finding_cannot_cite_source_outside_query_evidence():
    class InvalidCitationModel(FakeResearchModel):
        def extract_findings(self, query, evidence):
            return [
                ResearchFinding(
                    insight="Unsupported claim.",
                    source_refs=["CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"],
                )
            ]

    with pytest.raises(ValueError, match="outside retrieved evidence"):
        run_claim_research(
            store=ClaimStore(SAMPLE_OUTPUT),
            question="What repairs were invoiced?",
            model=InvalidCitationModel(),
            max_depth=1,
        )


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="question cannot be empty"):
        run_claim_research(ClaimStore(SAMPLE_OUTPUT), "  ", FakeResearchModel())


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"queries_per_question": 0}, "queries_per_question must be at least 1"),
        ({"max_depth": 0}, "max_depth must be at least 1"),
        ({"top_k": 0}, "top_k must be at least 1"),
    ],
)
def test_numeric_limits_are_rejected(argument, message):
    with pytest.raises(ValueError, match=message):
        run_claim_research(
            store=ClaimStore(SAMPLE_OUTPUT),
            question="What repairs were invoiced?",
            model=FakeResearchModel(),
            **argument,
        )


def test_info_logging_traces_research_without_ocr_text(caplog):
    with caplog.at_level(logging.INFO, logger="knowledge_agent.research.agent"):
        run_claim_research(
            store=ClaimStore(SAMPLE_OUTPUT),
            question="What repairs were invoiced?",
            model=FakeResearchModel(),
            max_depth=1,
        )

    assert "research_start" in caplog.text
    assert "research_layer layer=1" in caplog.text
    assert "queries=['repair invoice']" in caplog.text
    assert f"source_refs=['{INVOICE_REF}']" in caplog.text
    assert "research_complete findings=1 sources=1" in caplog.text
    assert "Labor: 3.0 hours" not in caplog.text
    assert "Parts: front bumper cover" not in caplog.text
    assert "secret-test-key" not in caplog.text
