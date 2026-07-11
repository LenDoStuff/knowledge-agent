import pytest
from pydantic import ValidationError

from knowledge_agent.agents.claim_researcher.models import (
    ChatMessage,
    DraftAnswer,
    EvidenceItem,
    GapReview,
    ResearchAuditTrail,
    ResearchAnswer,
    ResearchFinding,
    ResearchLlmAuditEntry,
    ResearchPlan,
    ResearchQuery,
    ResearchSearch,
    ResearchToolAuditEntry,
)


SOURCE_REF = "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"


def test_research_models_capture_plan_searches_and_conversation():
    message = ChatMessage(role="user", content="When was the loss?")
    query = ResearchQuery(query="loss date", research_goal="Find the loss date")
    plan = ResearchPlan(objectives=["Establish the loss date."], queries=[query])
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
        question=message.content,
        answer=f"The loss was June 1, 2026. [{SOURCE_REF}]",
        plan=plan,
        searches=[ResearchSearch(query=query, source_refs=[SOURCE_REF])],
        findings=[finding],
        source_refs=[SOURCE_REF],
    )

    assert evidence.source_ref == SOURCE_REF
    assert answer.plan == plan
    assert answer.searches[0].query == query


def test_research_models_reject_empty_contract_fields():
    query = ResearchQuery(query="loss date", research_goal="Find the loss date")
    finding = ResearchFinding(insight="A fact", source_refs=[SOURCE_REF])

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=" ")
    with pytest.raises(ValidationError):
        ResearchPlan(objectives=[], queries=[query])
    with pytest.raises(ValidationError):
        ResearchPlan(objectives=["Find the date"], queries=[])
    with pytest.raises(ValidationError):
        ResearchFinding(insight="A fact", source_refs=[])
    with pytest.raises(ValidationError):
        GapReview(
            complete=True,
            queries=[query],
        )
    with pytest.raises(ValidationError):
        ResearchAnswer(
            question="When was the loss?",
            answer="The loss was June 1, 2026.",
            plan=ResearchPlan(objectives=["Find date"], queries=[query]),
            searches=[],
            findings=[finding],
            source_refs=[],
        )


def test_draft_answer_allows_no_sources_for_insufficient_evidence():
    draft = DraftAnswer(
        answer="The claim knowledge base does not contain enough evidence."
    )
    assert draft.source_refs == []


def test_draft_answer_keeps_bracketed_source_references_exact():
    draft = DraftAnswer(
        answer=f"A supported answer. [{SOURCE_REF}]",
        source_refs=[f"[{SOURCE_REF}]"],
    )

    assert draft.source_refs == [f"[{SOURCE_REF}]"]


def test_audit_trail_round_trips_discriminated_entries():
    query = ResearchQuery(query="loss date", research_goal="Find the loss date")
    evidence = EvidenceItem(
        document_id="DOC-001",
        document_type="fnol",
        document_title="First Notice of Loss",
        page_ids=["CLM-SAMPLE-001:p1"],
        source_ref=SOURCE_REF,
        text="Loss date: 2026-06-01",
    )
    trail = ResearchAuditTrail(
        entries=[
            ResearchLlmAuditEntry(
                operation="plan_research",
                response_model="ResearchPlan",
                system_prompt="Plan the research.",
                user_prompt="When was the loss?",
                result={"objectives": ["Find the loss date."]},
            ),
            ResearchToolAuditEntry(
                query=query,
                top_k=8,
                result=[evidence],
            ),
        ]
    )

    restored = ResearchAuditTrail.model_validate(trail.model_dump(mode="json"))

    assert isinstance(restored.entries[0], ResearchLlmAuditEntry)
    assert isinstance(restored.entries[1], ResearchToolAuditEntry)
    assert restored.entries[1].result == [evidence]


@pytest.mark.parametrize(
    "entry",
    [
        {
            "kind": "llm",
            "operation": "plan_research",
            "response_model": "ResearchPlan",
            "system_prompt": "system",
            "user_prompt": "user",
        },
        {
            "kind": "llm",
            "operation": "plan_research",
            "response_model": "ResearchPlan",
            "system_prompt": "system",
            "user_prompt": "user",
            "result": {},
            "error": "failed",
        },
        {
            "kind": "tool",
            "query": {"query": "loss date", "research_goal": "Find it"},
            "top_k": 8,
        },
        {
            "kind": "tool",
            "query": {"query": "loss date", "research_goal": "Find it"},
            "top_k": 8,
            "result": [],
            "error": "failed",
        },
    ],
)
def test_audit_entry_requires_exactly_one_result_or_error(entry):
    with pytest.raises(ValidationError, match="exactly one result or error"):
        ResearchAuditTrail.model_validate({"entries": [entry]})
