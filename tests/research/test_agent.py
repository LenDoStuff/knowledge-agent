import logging
import re
from pathlib import Path
from threading import Barrier, Lock, get_ident
from typing import Callable

import pytest

from knowledge_agent.agents.claim_researcher import run_claim_research
from knowledge_agent.agents.claim_researcher.llm import FindingSet
from knowledge_agent.agents.claim_researcher.models import (
    ChatMessage,
    DraftAnswer,
    GapReview,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
)
from knowledge_agent.claims.store import load_claim_store


SAMPLE_OUTPUT = Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
INVOICE_REF = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
FNOL_REF = "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"


def build_research_parser(
    *,
    plan: ResearchPlan | None = None,
    findings_by_query: dict[str, list[ResearchFinding]] | None = None,
    gap_reviews: list[GapReview] | None = None,
    draft: DraftAnswer | None = None,
    on_extract: Callable[[str], None] | None = None,
):
    selected_plan = plan or _default_plan()
    selected_findings = findings_by_query or _default_findings_by_query()
    queued_gap_reviews = list(gap_reviews) if gap_reviews is not None else None
    lock = Lock()
    state = {
        "plan_prompts": [],
        "extraction_prompts": [],
        "review_prompts": [],
        "review_query_limits": [],
        "answer_prompts": [],
    }

    def parse(system, user, response_model):
        if response_model is ResearchPlan:
            with lock:
                state["plan_prompts"].append(user)
            return selected_plan
        if response_model is FindingSet:
            query = _query_from_extract_prompt(user)
            with lock:
                state["extraction_prompts"].append((query, user))
            if on_extract is not None:
                on_extract(query)
            return FindingSet(
                findings=[
                    finding.model_copy(deep=True)
                    for finding in selected_findings.get(query, [])
                ]
            )
        if response_model is GapReview:
            with lock:
                state["review_prompts"].append(user)
                state["review_query_limits"].append(_query_limit_from_prompt(user))
            if queued_gap_reviews is not None:
                return queued_gap_reviews.pop(0)
            return _default_gap_review(user)
        if response_model is DraftAnswer:
            with lock:
                state["answer_prompts"].append(user)
            return draft or _default_draft(user)
        raise AssertionError(f"Unexpected response model: {response_model}")

    return parse, state


def _default_plan() -> ResearchPlan:
    return ResearchPlan(
        objectives=["Identify the invoiced work and its support."],
        queries=[
            ResearchQuery(
                query="repair invoice",
                research_goal="Identify invoiced repair work.",
            )
        ],
    )


def _default_findings_by_query() -> dict[str, list[ResearchFinding]]:
    invoice_finding = ResearchFinding(
        insight="The invoice lists labor and a front bumper cover.",
        source_refs=[INVOICE_REF],
    )
    return {
        "repair invoice": [invoice_finding, invoice_finding.model_copy(deep=True)],
        "loss date collision": [
            ResearchFinding(
                insight="The collision occurred on June 1, 2026.",
                source_refs=[FNOL_REF],
            )
        ],
    }


def _default_gap_review(user_prompt: str) -> GapReview:
    if FNOL_REF in _validated_findings_context(user_prompt):
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


def _default_draft(user_prompt: str) -> DraftAnswer:
    findings_context = _validated_findings_context(user_prompt)
    refs = []
    if INVOICE_REF in findings_context:
        refs.append(INVOICE_REF)
    if FNOL_REF in findings_context:
        refs.append(FNOL_REF)
    if not refs:
        return DraftAnswer(
            answer="The claim knowledge base does not contain enough evidence.",
            source_refs=[],
        )
    citations = " ".join(f"[{source_ref}]" for source_ref in refs)
    return DraftAnswer(
        answer=f"The claim contains supported repair details. {citations}",
        source_refs=refs,
    )


def _query_from_extract_prompt(user_prompt: str) -> str:
    match = re.search(r"^Search query: (.+)$", user_prompt, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"Could not find query in prompt: {user_prompt}")
    return match.group(1)


def _validated_findings_context(user_prompt: str) -> str:
    match = re.search(
        r"Validated findings:\n(.+?)\n\n(?:Available documents:|Write a concise)",
        user_prompt,
        flags=re.DOTALL,
    )
    if match is None:
        return ""
    return match.group(1)


def _query_limit_from_prompt(user_prompt: str) -> int:
    match = re.search(r"Return at most (\d+) new queries", user_prompt)
    if match is None:
        raise AssertionError(f"Could not find query limit in prompt: {user_prompt}")
    return int(match.group(1))


def test_run_claim_research_uses_history_manifest_and_retrieved_evidence():
    parse, state = build_research_parser()
    observed_steps = []
    history = [
        ChatMessage(role="user", content="Was there a collision?"),
        ChatMessage(role="assistant", content=f"Yes. [{FNOL_REF}]"),
    ]

    answer = run_claim_research(
        store=load_claim_store(SAMPLE_OUTPUT),
        question="What repairs were invoiced?",
        parse_structured_output=parse,
        max_depth=1,
        history=history,
        on_step=observed_steps.append,
    )

    plan_prompt = state["plan_prompts"][0]
    assert "Was there a collision?" in plan_prompt
    assert f"Yes. [{FNOL_REF}]" in plan_prompt
    assert "DOC-001" in plan_prompt
    assert "DOC-002" in plan_prompt
    _, extraction_prompt = state["extraction_prompts"][0]
    assert INVOICE_REF in extraction_prompt
    assert "front bumper cover" in extraction_prompt
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
    parse, state = build_research_parser()

    answer = run_claim_research(
        store=load_claim_store(SAMPLE_OUTPUT),
        question="What repairs were invoiced?",
        parse_structured_output=parse,
        queries_per_question=4,
        max_depth=3,
    )

    assert [search.query.query for search in answer.searches] == [
        "repair invoice",
        "loss date collision",
    ]
    assert state["review_query_limits"] == [2, 1]
    assert len(answer.gap_reviews) == 2
    assert [step.stage for step in answer.steps].count("gap_review") == 2
    assert len(answer.findings) == 2
    assert answer.source_refs == [INVOICE_REF, FNOL_REF]


def test_complete_gap_review_stops_without_another_search():
    parse, state = build_research_parser(gap_reviews=[GapReview(complete=True)])

    answer = run_claim_research(
        load_claim_store(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        parse,
        max_depth=3,
    )

    assert len(state["review_prompts"]) == 1
    assert len(answer.searches) == 1


def test_research_queries_run_concurrently_and_commit_in_plan_order():
    query_texts = ["alpha", "bravo", "charlie", "delta"]
    barrier = Barrier(len(query_texts))
    lock = Lock()
    worker_threads = set()

    def on_extract(_query):
        with lock:
            worker_threads.add(get_ident())
        barrier.wait(timeout=5)

    plan = ResearchPlan(
        objectives=["Test concurrent research."],
        queries=[
            ResearchQuery(query=value, research_goal=f"Search {value}.")
            for value in query_texts
        ],
    )
    parse, _ = build_research_parser(
        plan=plan,
        findings_by_query={},
        on_extract=on_extract,
    )
    callback_threads = []

    answer = run_claim_research(
        load_claim_store(SAMPLE_OUTPUT),
        "Run several searches.",
        parse,
        max_depth=1,
        on_step=lambda step: callback_threads.append(get_ident()),
    )

    assert len(worker_threads) == 4
    assert get_ident() not in worker_threads
    assert set(callback_threads) == {get_ident()}
    assert [search.query.query for search in answer.searches] == query_texts
    assert [step.query for step in answer.steps if step.stage == "tool"] == query_texts


def test_duplicate_queries_and_findings_are_executed_once():
    default_plan = _default_plan()
    plan = default_plan.model_copy(update={"queries": [default_plan.queries[0]] * 2})
    parse, state = build_research_parser(plan=plan)

    answer = run_claim_research(
        load_claim_store(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        parse,
        max_depth=1,
    )

    assert len(state["extraction_prompts"]) == 1
    assert len(answer.searches) == 1
    assert len(answer.findings) == 1


def test_finding_cannot_cite_source_outside_query_evidence():
    parse, _ = build_research_parser(
        findings_by_query={
            "repair invoice": [
                ResearchFinding(
                    insight="Unsupported claim.",
                    source_refs=["CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"],
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="outside retrieved evidence"):
        run_claim_research(
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            parse,
            max_depth=1,
        )


def test_page_suffix_on_finding_source_is_rejected():
    parse, _ = build_research_parser(
        findings_by_query={
            "repair invoice": [
                ResearchFinding(
                    insight="The invoice lists repairs.",
                    source_refs=[f"{INVOICE_REF}:p2"],
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="outside retrieved evidence"):
        run_claim_research(
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            parse,
            max_depth=1,
        )


@pytest.mark.parametrize("failure", ["unsupported", "undeclared"])
def test_final_answer_citations_must_be_supported_and_declared(failure):
    if failure == "unsupported":
        invalid = "CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"
        draft = DraftAnswer(answer=f"Unsupported. [{invalid}]", source_refs=[invalid])
    else:
        draft = DraftAnswer(
            answer=f"A repair was found. [{INVOICE_REF}]",
            source_refs=[],
        )
    parse, _ = build_research_parser(draft=draft)

    expected = (
        "outside validated findings"
        if failure == "unsupported"
        else "undeclared source references"
    )
    with pytest.raises(ValueError, match=expected):
        run_claim_research(
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            parse,
            max_depth=1,
        )


def test_unused_declared_sources_are_not_displayed_as_answer_citations():
    parse, _ = build_research_parser(
        draft=DraftAnswer(
            answer=f"The invoice lists repairs. [{INVOICE_REF}]",
            source_refs=[INVOICE_REF, FNOL_REF],
        )
    )

    answer = run_claim_research(
        load_claim_store(SAMPLE_OUTPUT),
        "What repairs were invoiced?",
        parse,
        max_depth=2,
    )

    assert answer.source_refs == [INVOICE_REF]


def test_page_suffix_on_answer_source_is_rejected():
    decorated = f"{INVOICE_REF}:p2"
    parse, _ = build_research_parser(
        draft=DraftAnswer(
            answer=f"The invoice lists repairs. [{decorated}]",
            source_refs=[decorated],
        )
    )

    with pytest.raises(ValueError, match="outside validated findings"):
        run_claim_research(
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            parse,
            max_depth=1,
        )


def test_empty_question_is_rejected():
    parse, _ = build_research_parser()
    with pytest.raises(ValueError, match="question cannot be empty"):
        run_claim_research(load_claim_store(SAMPLE_OUTPUT), "  ", parse)


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"queries_per_question": 0}, "queries_per_question must be at least 1"),
        ({"max_depth": 0}, "max_depth must be at least 1"),
        ({"top_k": 0}, "top_k must be at least 1"),
    ],
)
def test_numeric_limits_are_rejected(argument, message):
    parse, _ = build_research_parser()
    with pytest.raises(ValueError, match=message):
        run_claim_research(
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            parse,
            **argument,
        )


def test_info_logging_traces_research_without_ocr_text(caplog):
    parse, _ = build_research_parser()
    with caplog.at_level(
        logging.INFO,
        logger="knowledge_agent.agents.claim_researcher.workflow",
    ):
        run_claim_research(
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
            parse,
            max_depth=1,
        )

    assert "research_start" in caplog.text
    assert "research_layer layer=1" in caplog.text
    assert "query='repair invoice'" in caplog.text
    assert f"source_refs=['{INVOICE_REF}']" in caplog.text
    assert "research_complete searches=1 evidence=1 findings=1 sources=1" in caplog.text
    assert "Labor: 3.0 hours" not in caplog.text
    assert "Parts: front bumper cover" not in caplog.text
