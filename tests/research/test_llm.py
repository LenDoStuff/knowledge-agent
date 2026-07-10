import logging
from pathlib import Path

from knowledge_agent.agents.claim_researcher.llm import (
    FindingSet,
    extract_findings,
    plan_research,
    review_gaps,
    write_answer,
)
from knowledge_agent.agents.claim_researcher.models import (
    ChatMessage,
    DraftAnswer,
    EvidenceItem,
    GapReview,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
)
from knowledge_agent.claims.store import load_claim_store


SAMPLE_OUTPUT = Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
SOURCE_REF = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"


def build_parser(outputs):
    queued_outputs = list(outputs)
    calls = []

    def parse(system, user, response_model):
        calls.append((system, user, response_model))
        return queued_outputs.pop(0)

    return parse, calls


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
    parse, calls = build_parser(
        [
            plan,
            FindingSet(findings=[finding]),
            review,
            draft,
        ]
    )
    documents = load_claim_store(SAMPLE_OUTPUT).documents
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

    assert plan_research(parse, "What was repaired?", history, documents, 4) == plan
    assert extract_findings(parse, query, evidence) == [finding]
    assert (
        review_gaps(parse, "What was repaired?", history, plan, [finding], documents, 2)
        == review
    )
    assert write_answer(parse, "What was repaired?", history, plan, [finding]) == draft
    assert [call[2] for call in calls] == [
        ResearchPlan,
        FindingSet,
        GapReview,
        DraftAnswer,
    ]
    assert "Was the vehicle damaged?" in calls[0][1]
    assert "Was the vehicle damaged?" in calls[2][1]
    assert "Was the vehicle damaged?" in calls[3][1]
    assert "Identify the repaired parts." in calls[3][1]
    assert SOURCE_REF in calls[3][1]


def test_debug_logging_contains_exact_prompt_and_parsed_output(caplog):
    query = ResearchQuery(
        query="repair invoice",
        research_goal="Identify invoiced repairs.",
    )
    plan = ResearchPlan(objectives=["Identify repairs."], queries=[query])
    parse, _ = build_parser([plan])
    documents = load_claim_store(SAMPLE_OUTPUT).documents

    with caplog.at_level(
        logging.DEBUG,
        logger="knowledge_agent.agents.claim_researcher.llm",
    ):
        plan_research(parse, "What was repaired?", [], documents, 1)

    assert "research_llm_prompt operation=plan_research" in caplog.text
    assert "Available documents:" in caplog.text
    assert "Repair Invoice" in caplog.text
    assert "research_llm_output operation=plan_research" in caplog.text
    assert '"query":"repair invoice"' in caplog.text
    assert "secret-test-key" not in caplog.text
