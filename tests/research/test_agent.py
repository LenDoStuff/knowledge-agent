import logging
from pathlib import Path

import pytest

from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.research.agent import run_claim_research
from knowledge_agent.research.models import (
    ChatMessage,
    DraftAnswer,
    GapReview,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
)


SAMPLE_OUTPUT = Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
INVOICE_REF = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
FNOL_REF = "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"


class FakeResearchModel:
    def __init__(self) -> None:
        self.plan_calls = []
        self.extraction_calls = []
        self.review_calls = []
        self.answer_calls = []

    def plan_research(self, question, history, documents, queries_per_question):
        self.plan_calls.append((question, history, documents, queries_per_question))
        return ResearchPlan(
            objectives=["Identify the invoiced work and its support."],
            queries=[
                ResearchQuery(
                    query="repair invoice",
                    research_goal="Identify invoiced repair work.",
                )
            ],
        )

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
        )
        return [finding, finding.model_copy()]

    def review_gaps(
        self,
        question,
        history,
        plan,
        findings,
        documents,
        query_limit,
    ):
        self.review_calls.append(
            (question, history, plan, findings, documents, query_limit)
        )
        if any(FNOL_REF in finding.source_refs for finding in findings):
            return GapReview(complete=True)
        return GapReview(
            complete=False,
            missing_information=["The loss date is not established."],
            queries=[
                ResearchQuery(
                    query="loss date collision",
                    research_goal="Find the collision date.",
                )
            ],
        )

    def write_answer(self, question, history, plan, findings):
        self.answer_calls.append((question, history, plan, findings))
        refs = []
        if any(INVOICE_REF in finding.source_refs for finding in findings):
            refs.append(INVOICE_REF)
        if any(FNOL_REF in finding.source_refs for finding in findings):
            refs.append(FNOL_REF)
        citations = " ".join(f"[{source_ref}]" for source_ref in refs)
        return DraftAnswer(
            answer=f"The claim contains supported repair details. {citations}",
            source_refs=refs,
        )


def test_run_claim_research_uses_history_manifest_and_retrieved_evidence():
    model = FakeResearchModel()
    observed_steps = []
    history = [
        ChatMessage(role="user", content="Was there a collision?"),
        ChatMessage(role="assistant", content=f"Yes. [{FNOL_REF}]"),
    ]

    answer = run_claim_research(
        store=ClaimStore(SAMPLE_OUTPUT),
        question="What repairs were invoiced?",
        model=model,
        max_depth=1,
        history=history,
        on_step=observed_steps.append,
    )

    assert model.plan_calls[0][1] == history
    assert [document.id for document in model.plan_calls[0][2]] == [
        "DOC-001",
        "DOC-002",
    ]
    evidence = model.extraction_calls[0][1]
    assert evidence[0].source_ref == INVOICE_REF
    assert "front bumper cover" in evidence[0].text
    assert answer.source_refs == [INVOICE_REF]
    assert len(answer.findings) == 1
    assert answer.searches[0].source_refs == [INVOICE_REF]
    assert answer.plan.objectives
    assert [step.stage for step in answer.steps] == [
        "plan",
        "tool",
        "validation",
        "answer",
    ]
    assert observed_steps == answer.steps
    assert answer.steps[1].tool_name == "claim_search"
    assert answer.gap_reviews == []


def test_gap_review_adds_a_smaller_second_research_layer():
    model = FakeResearchModel()

    answer = run_claim_research(
        store=ClaimStore(SAMPLE_OUTPUT),
        question="What repairs were invoiced?",
        model=model,
        queries_per_question=4,
        max_depth=3,
    )

    assert [search.query.query for search in answer.searches] == [
        "repair invoice",
        "loss date collision",
    ]
    assert [call[-1] for call in model.review_calls] == [2, 1]
    assert len(answer.gap_reviews) == 2
    assert [step.stage for step in answer.steps].count("gap_review") == 2
    assert len(answer.findings) == 2
    assert answer.source_refs == [INVOICE_REF, FNOL_REF]


def test_complete_gap_review_stops_without_another_search():
    class CompleteModel(FakeResearchModel):
        def review_gaps(self, *args):
            self.review_calls.append(args)
            return GapReview(complete=True)

    model = CompleteModel()
    answer = run_claim_research(
        ClaimStore(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        model,
        max_depth=3,
    )

    assert len(model.review_calls) == 1
    assert len(answer.searches) == 1


def test_duplicate_queries_and_findings_are_executed_once():
    class DuplicateModel(FakeResearchModel):
        def plan_research(self, question, history, documents, query_count):
            plan = super().plan_research(question, history, documents, query_count)
            return plan.model_copy(update={"queries": [plan.queries[0]] * 2})

    model = DuplicateModel()
    answer = run_claim_research(
        ClaimStore(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        model,
        max_depth=1,
    )

    assert len(model.extraction_calls) == 1
    assert len(answer.searches) == 1
    assert len(answer.findings) == 1


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
            ClaimStore(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            InvalidCitationModel(),
            max_depth=1,
        )


def test_page_suffix_is_removed_from_an_exact_retrieved_source():
    class PageDecoratedCitationModel(FakeResearchModel):
        def extract_findings(self, query, evidence):
            return [
                ResearchFinding(
                    insight="The invoice lists repairs.",
                    source_refs=[f"{INVOICE_REF}:p2"],
                )
            ]

    answer = run_claim_research(
        ClaimStore(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        PageDecoratedCitationModel(),
        max_depth=1,
    )

    assert answer.findings[0].source_refs == [INVOICE_REF]
    assert answer.source_refs == [INVOICE_REF]


@pytest.mark.parametrize("failure", ["unsupported", "undeclared"])
def test_final_answer_citations_must_be_supported_and_declared(failure):
    class InvalidAnswerModel(FakeResearchModel):
        def write_answer(self, question, history, plan, findings):
            if failure == "unsupported":
                invalid = "CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"
                return DraftAnswer(answer=f"Unsupported. [{invalid}]", source_refs=[invalid])
            return DraftAnswer(
                answer=f"A repair was found. [{INVOICE_REF}]",
                source_refs=[],
            )

    expected = (
        "outside validated findings"
        if failure == "unsupported"
        else "undeclared source references"
    )
    with pytest.raises(ValueError, match=expected):
        run_claim_research(
            ClaimStore(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            InvalidAnswerModel(),
            max_depth=1,
        )


def test_unused_declared_sources_are_not_displayed_as_answer_citations():
    class ExtraDeclaredSourceModel(FakeResearchModel):
        def write_answer(self, question, history, plan, findings):
            return DraftAnswer(
                answer=f"The invoice lists repairs. [{INVOICE_REF}]",
                source_refs=[INVOICE_REF, FNOL_REF],
            )

    answer = run_claim_research(
        ClaimStore(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        ExtraDeclaredSourceModel(),
        max_depth=2,
    )

    assert answer.source_refs == [INVOICE_REF]


def test_page_suffix_is_removed_from_answer_citations():
    class PageDecoratedAnswerModel(FakeResearchModel):
        def write_answer(self, question, history, plan, findings):
            decorated = f"{INVOICE_REF}:p2"
            return DraftAnswer(
                answer=f"The invoice lists repairs. [{decorated}]",
                source_refs=[decorated],
            )

    answer = run_claim_research(
        ClaimStore(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        PageDecoratedAnswerModel(),
        max_depth=1,
    )

    assert answer.answer.endswith(f"[{INVOICE_REF}]")
    assert answer.source_refs == [INVOICE_REF]


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
            ClaimStore(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            FakeResearchModel(),
            **argument,
        )


def test_info_logging_traces_research_without_ocr_text(caplog):
    with caplog.at_level(logging.INFO, logger="knowledge_agent.research.agent"):
        run_claim_research(
            ClaimStore(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            FakeResearchModel(),
            max_depth=1,
        )

    assert "research_start" in caplog.text
    assert "research_layer layer=1" in caplog.text
    assert "query='repair invoice'" in caplog.text
    assert f"source_refs=['{INVOICE_REF}']" in caplog.text
    assert "research_complete searches=1 evidence=1 findings=1 sources=1" in caplog.text
    assert "Labor: 3.0 hours" not in caplog.text
    assert "Parts: front bumper cover" not in caplog.text
