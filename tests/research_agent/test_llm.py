from pathlib import Path
import logging

from claim_kb import ClaimKbKnowledgeStore
from research_agent.llm import (
    FindingSet,
    QueryPlan,
    ResearchResponsesLlm,
    WrittenAnswer,
)
from research_agent.schemas import EvidenceItem, ResearchFinding, ResearchQuery


SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claim_kb" / "sample_output"
)
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
    return ResearchResponsesLlm(client), client


def test_research_llm_uses_structured_responses_for_all_steps():
    query = ResearchQuery(
        query="repair invoice",
        research_goal="Identify invoiced repairs.",
    )
    finding = ResearchFinding(
        insight="The invoice lists a front bumper cover.",
        source_refs=[SOURCE_REF],
    )
    llm, client = build_llm(
        [
            QueryPlan(queries=[query]),
            FindingSet(findings=[finding]),
            WrittenAnswer(answer=f"A bumper cover was invoiced. [{SOURCE_REF}]"),
        ]
    )
    documents = ClaimKbKnowledgeStore(SAMPLE_OUTPUT).manifest.documents
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

    assert llm.plan_queries("What was repaired?", documents, 4) == [query]
    assert llm.extract_findings(query, evidence) == [finding]
    assert SOURCE_REF in llm.write_answer("What was repaired?", [finding])
    assert [call[2] for call in client.calls] == [
        QueryPlan,
        FindingSet,
        WrittenAnswer,
    ]


def test_debug_logging_contains_exact_prompt_and_parsed_output(caplog):
    query = ResearchQuery(
        query="repair invoice",
        research_goal="Identify invoiced repairs.",
    )
    llm, _ = build_llm([QueryPlan(queries=[query])])
    documents = ClaimKbKnowledgeStore(SAMPLE_OUTPUT).manifest.documents

    with caplog.at_level(logging.DEBUG, logger="research_agent.llm"):
        llm.plan_queries("What was repaired?", documents, 1)

    assert "research_llm_prompt operation=plan_queries" in caplog.text
    assert "Available documents:" in caplog.text
    assert "Repair Invoice" in caplog.text
    assert "research_llm_output operation=plan_queries" in caplog.text
    assert '"query":"repair invoice"' in caplog.text
    assert "secret-test-key" not in caplog.text
