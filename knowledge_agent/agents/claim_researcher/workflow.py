"""Typed, bounded research workflow over one persisted claim."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from knowledge_agent.agents.claim_researcher.llm import (
    extract_findings,
    plan_research,
    review_gaps,
    write_answer,
)
from knowledge_agent.agents.claim_researcher.models import (
    ChatMessage,
    ResearchAuditEntry,
    ResearchAnswer,
    ResearchLlmAuditEntry,
    ResearchQuery,
    ResearchSearch,
    ResearchStep,
    ResearchToolAuditEntry,
)
from knowledge_agent.agents.claim_researcher.state import ResearchState, _QueryResult
from knowledge_agent.agents.claim_researcher.tools import claim_search
from knowledge_agent.agents.claim_researcher.validation import (
    deduplicate_findings,
    unique_queries,
    validate_draft,
    validate_finding_sources,
)
from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.llm.client import StructuredOutputParser


LOGGER = logging.getLogger(__name__)
_MAX_CONCURRENT_WORKERS = 4
ResearchLlmOperation = Literal[
    "plan_research",
    "extract_findings",
    "review_gaps",
    "write_answer",
]


def run_claim_research(
    store: ClaimStore,
    question: str,
    parse_structured_output: StructuredOutputParser,
    queries_per_question: int = 4,
    max_depth: int = 2,
    top_k: int = 8,
    history: Sequence[ChatMessage] = (),
    on_step: Callable[[ResearchStep], None] | None = None,
    on_audit: Callable[[ResearchAuditEntry], None] | None = None,
) -> ResearchAnswer:
    question = _validate_research_request(
        question,
        queries_per_question,
        max_depth,
        top_k,
    )
    conversation = list(history)
    LOGGER.info(
        "research_start question=%r history=%d queries_per_question=%d "
        "max_depth=%d top_k=%d",
        question,
        len(conversation),
        queries_per_question,
        max_depth,
        top_k,
    )
    state = _start_research(
        store,
        question,
        conversation,
        parse_structured_output,
        queries_per_question,
        on_step,
        on_audit,
    )
    _run_research_layers(
        store,
        question,
        conversation,
        parse_structured_output,
        state,
        queries_per_question,
        max_depth,
        top_k,
        on_step,
        on_audit,
    )
    return _finish_research(
        question,
        conversation,
        parse_structured_output,
        state,
        on_step,
        on_audit,
    )


def _validate_research_request(
    question: str,
    queries_per_question: int,
    max_depth: int,
    top_k: int,
) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if queries_per_question < 1:
        raise ValueError("queries_per_question must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return question


def _start_research(
    store: ClaimStore,
    question: str,
    conversation: list[ChatMessage],
    parse_structured_output: StructuredOutputParser,
    queries_per_question: int,
    on_step: Callable[[ResearchStep], None] | None,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> ResearchState:
    plan = plan_research(
        _audited_parser(parse_structured_output, "plan_research", on_audit),
        question,
        conversation,
        store.documents,
        queries_per_question,
    )
    state = ResearchState(plan=plan)
    _record_step(
        state,
        ResearchStep(
            stage="plan",
            message=(
                f"Planned {len(plan.queries)} searches for "
                f"{len(plan.objectives)} answer objectives."
            ),
        ),
        on_step,
    )
    return state


def _run_research_layers(
    store: ClaimStore,
    question: str,
    conversation: list[ChatMessage],
    parse_structured_output: StructuredOutputParser,
    state: ResearchState,
    queries_per_question: int,
    max_depth: int,
    top_k: int,
    on_step: Callable[[ResearchStep], None] | None,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> None:
    pending_queries = unique_queries(state.plan.queries)[:queries_per_question]
    seen_queries: set[str] = set()
    layer_query_count = queries_per_question

    for layer_index in range(max_depth):
        LOGGER.info(
            "research_layer layer=%d queries=%d objectives=%d",
            layer_index + 1,
            len(pending_queries),
            len(state.plan.objectives),
        )
        layer_queries = _take_unseen_queries(pending_queries, seen_queries)
        query_results = _execute_queries(
            store,
            layer_queries,
            parse_structured_output,
            top_k,
            on_audit,
        )
        _commit_query_results(
            state,
            query_results,
            layer_index + 1,
            on_step,
            on_audit,
        )
        if layer_index == max_depth - 1:
            break

        layer_query_count = max(1, layer_query_count // 2)
        pending_queries = _plan_follow_up_queries(
            store,
            question,
            conversation,
            parse_structured_output,
            state,
            seen_queries,
            layer_query_count,
            layer_index + 1,
            on_step,
            on_audit,
        )
        if not pending_queries:
            break


def _take_unseen_queries(
    queries: list[ResearchQuery],
    seen_queries: set[str],
) -> list[ResearchQuery]:
    layer_queries = []
    for query in queries:
        query_key = query.query.casefold()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        layer_queries.append(query)
    return layer_queries


def _execute_queries(
    store: ClaimStore,
    queries: list[ResearchQuery],
    parse_structured_output: StructuredOutputParser,
    top_k: int,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> list[_QueryResult]:
    def execute_query(query: ResearchQuery) -> _QueryResult:
        audit_entries: list[ResearchAuditEntry] = []
        capture = audit_entries.append if on_audit is not None else None
        try:
            evidence = claim_search(store, query, top_k)
        except Exception as exc:
            if capture is not None:
                capture(
                    ResearchToolAuditEntry(
                        query=query,
                        top_k=top_k,
                        error=_error_message(exc),
                    )
                )
            return _QueryResult(
                query=query,
                evidence=[],
                findings=[],
                audit_entries=audit_entries,
                error=exc,
            )

        if capture is not None:
            capture(
                ResearchToolAuditEntry(
                    query=query,
                    top_k=top_k,
                    result=evidence,
                )
            )
        try:
            findings = extract_findings(
                _audited_parser(
                    parse_structured_output,
                    "extract_findings",
                    capture,
                ),
                query,
                evidence,
            )
        except Exception as exc:
            return _QueryResult(
                query=query,
                evidence=evidence,
                findings=[],
                audit_entries=audit_entries,
                error=exc,
            )
        return _QueryResult(
            query=query,
            evidence=evidence,
            findings=findings,
            audit_entries=audit_entries,
        )

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_WORKERS) as executor:
        return list(executor.map(execute_query, queries))


def _commit_query_results(
    state: ResearchState,
    query_results: list[_QueryResult],
    layer_number: int,
    on_step: Callable[[ResearchStep], None] | None,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> None:
    if on_audit is not None:
        for result in query_results:
            for entry in result.audit_entries:
                on_audit(entry)

    for result in query_results:
        if result.error is not None:
            raise result.error
        validate_finding_sources(result.findings, result.evidence)
        for item in result.evidence:
            state.evidence_by_ref.setdefault(item.source_ref, item)
        source_refs = [item.source_ref for item in result.evidence]
        state.searches.append(
            ResearchSearch(query=result.query, source_refs=source_refs)
        )
        _record_step(
            state,
            ResearchStep(
                stage="tool",
                message=(
                    f"claim_search returned {len(result.evidence)} evidence chunks "
                    f"for {result.query.query!r}."
                ),
                tool_name="claim_search",
                query=result.query.query,
                source_refs=source_refs,
            ),
            on_step,
        )
        LOGGER.info(
            "research_evidence layer=%d query=%r count=%d source_refs=%s",
            layer_number,
            result.query.query,
            len(result.evidence),
            source_refs,
        )
        state.findings.extend(result.findings)

    state.findings = deduplicate_findings(state.findings)
    _record_step(
        state,
        ResearchStep(
            stage="validation",
            message=f"Validated {len(state.findings)} supported findings.",
        ),
        on_step,
    )


def _plan_follow_up_queries(
    store: ClaimStore,
    question: str,
    conversation: list[ChatMessage],
    parse_structured_output: StructuredOutputParser,
    state: ResearchState,
    seen_queries: set[str],
    query_limit: int,
    layer_number: int,
    on_step: Callable[[ResearchStep], None] | None,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> list[ResearchQuery]:
    review = review_gaps(
        _audited_parser(parse_structured_output, "review_gaps", on_audit),
        question,
        conversation,
        state.plan,
        state.findings,
        store.documents,
        query_limit,
    )
    state.gap_reviews.append(review)
    message = "Evidence coverage is complete."
    if not review.complete:
        message = (
            f"Found {len(review.missing_information)} evidence gaps "
            f"and planned {len(review.queries)} follow-up searches."
        )
    _record_step(
        state,
        ResearchStep(stage="gap_review", message=message),
        on_step,
    )
    LOGGER.info(
        "research_gap_review layer=%d complete=%s gaps=%d queries=%d",
        layer_number,
        review.complete,
        len(review.missing_information),
        len(review.queries),
    )
    if review.complete:
        return []
    return [
        query
        for query in unique_queries(review.queries)
        if query.query.casefold() not in seen_queries
    ][:query_limit]


def _finish_research(
    question: str,
    conversation: list[ChatMessage],
    parse_structured_output: StructuredOutputParser,
    state: ResearchState,
    on_step: Callable[[ResearchStep], None] | None,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> ResearchAnswer:
    draft = write_answer(
        _audited_parser(parse_structured_output, "write_answer", on_audit),
        question,
        conversation,
        state.plan,
        state.findings,
    )
    answer_text, answer_sources = validate_draft(
        draft.answer,
        draft.source_refs,
        state.findings,
    )
    _record_step(
        state,
        ResearchStep(
            stage="answer",
            message=f"Wrote the answer with {len(answer_sources)} citations.",
            source_refs=answer_sources,
        ),
        on_step,
    )
    LOGGER.info(
        "research_complete searches=%d evidence=%d findings=%d sources=%d",
        len(state.searches),
        len(state.evidence_by_ref),
        len(state.findings),
        len(answer_sources),
    )
    return ResearchAnswer(
        question=question,
        answer=answer_text,
        plan=state.plan,
        searches=state.searches,
        gap_reviews=state.gap_reviews,
        steps=state.steps,
        findings=state.findings,
        source_refs=answer_sources,
    )


def _record_step(
    state: ResearchState,
    step: ResearchStep,
    on_step: Callable[[ResearchStep], None] | None,
) -> None:
    state.steps.append(step)
    if on_step is not None:
        on_step(step)


def _audited_parser(
    parse_structured_output: StructuredOutputParser,
    operation: ResearchLlmOperation,
    on_audit: Callable[[ResearchAuditEntry], None] | None,
) -> StructuredOutputParser:
    if on_audit is None:
        return parse_structured_output

    def parse(system: str, user: str, response_model):
        try:
            parsed = parse_structured_output(system, user, response_model)
        except Exception as exc:
            on_audit(
                ResearchLlmAuditEntry(
                    operation=operation,
                    response_model=response_model.__name__,
                    system_prompt=system,
                    user_prompt=user,
                    error=_error_message(exc),
                )
            )
            raise
        on_audit(
            ResearchLlmAuditEntry(
                operation=operation,
                response_model=response_model.__name__,
                system_prompt=system,
                user_prompt=user,
                result=parsed.model_dump(mode="json"),
            )
        )
        return parsed

    return parse


def _error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
