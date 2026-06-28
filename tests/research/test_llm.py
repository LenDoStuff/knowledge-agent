import logging
from pathlib import Path

from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.research.llm import FindingSet, ResponsesResearchModel
from knowledge_agent.research.models import (
    ChatMessage,
    DraftAnswer,
    EvidenceItem,
    GapReview,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
)


SAMPLE_OUTPUT = Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
SOURCE_REF = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"


class FakeStructuredOutputClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, system, user, response_model):
        self.calls.append((system, user, response_model))
        return self.outputs.pop(0)


def build_llm(outputs):
    client = FakeStructuredOutputClient(outputs)
    return ResponsesResearchModel(client), client


def test_research_llm_uses_structured_responses_for_every_stage():
    query = ResearchQuery(
        query="repair invoice",
        research_goal="Identify invoiced repairs.",
    )
    plan = ResearchPlan(
        objectives=["Identify the repaired parts."],
        queries=[query],
    )
    finding = ResearchFinding(
        insight="The invoice lists a front bumper cover.",
        source_refs=[SOURCE_REF],
    )
    review = GapReview(complete=True)
    draft = DraftAnswer(
        answer=f"A bumper cover was invoiced. [{SOURCE_REF}]",
        source_refs=[SOURCE_REF],
    )
    llm, client = build_llm(
        [
            plan,
            FindingSet(findings=[finding]),
            review,
            draft,
        ]
    )
    documents = ClaimStore(SAMPLE_OUTPUT).manifest.documents
    history = [ChatMessage(role="user", content="Was the vehicle damaged?")]
    evidence = [
        EvidenceItem(
            document_id="DOC-002",
            document_type="invoice",
            document_title="Repair Invoice",
            page_ids=["CLM-SAMPLE-001:p2"],
            source_ref=SOURCE_REF,
            text="Parts: front bumper cover",
        )
    ]

    assert llm.plan_research("What was repaired?", history, documents, 4) == plan
    assert llm.extract_findings(query, evidence) == [finding]
    assert (
        llm.review_gaps("What was repaired?", history, plan, [finding], documents, 2)
        == review
    )
    assert llm.write_answer("What was repaired?", history, plan, [finding]) == draft
    assert [call[2] for call in client.calls] == [
        ResearchPlan,
        FindingSet,
        GapReview,
        DraftAnswer,
    ]
    assert "Was the vehicle damaged?" in client.calls[0][1]
    assert "Was the vehicle damaged?" in client.calls[2][1]
    assert "Was the vehicle damaged?" in client.calls[3][1]
    assert "Identify the repaired parts." in client.calls[3][1]
    assert SOURCE_REF in client.calls[3][1]


def test_debug_logging_contains_exact_prompt_and_parsed_output(caplog):
    query = ResearchQuery(
        query="repair invoice",
        research_goal="Identify invoiced repairs.",
    )
    plan = ResearchPlan(objectives=["Identify repairs."], queries=[query])
    llm, _ = build_llm([plan])
    documents = ClaimStore(SAMPLE_OUTPUT).manifest.documents

    with caplog.at_level(logging.DEBUG, logger="knowledge_agent.research.llm"):
        llm.plan_research("What was repaired?", [], documents, 1)

    assert "research_llm_prompt operation=plan_research" in caplog.text
    assert "Available documents:" in caplog.text
    assert "Repair Invoice" in caplog.text
    assert "research_llm_output operation=plan_research" in caplog.text
    assert '"query":"repair invoice"' in caplog.text
    assert "secret-test-key" not in caplog.text
