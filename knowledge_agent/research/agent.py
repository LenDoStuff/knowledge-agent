"""Typed, bounded research pipeline over one persisted claim."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.research.llm import ResearchModel
from knowledge_agent.research.models import (
    ChatMessage,
    EvidenceItem,
    GapReview,
    ResearchAnswer,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
    ResearchSearch,
    ResearchStep,
)


LOGGER = logging.getLogger(__name__)
SOURCE_CITATION = re.compile(r"\[([^\[\]\s]+/[^\[\]\s]+#[^\[\]\s]+)\]")


@dataclass
class ResearchState:
    plan: ResearchPlan
    searches: list[ResearchSearch] = field(default_factory=list)
    gap_reviews: list[GapReview] = field(default_factory=list)
    steps: list[ResearchStep] = field(default_factory=list)
    evidence_by_ref: dict[str, EvidenceItem] = field(default_factory=dict)
    findings: list[ResearchFinding] = field(default_factory=list)


def run_claim_research(
    store: ClaimStore,
    question: str,
    model: ResearchModel,
    queries_per_question: int = 4,
    max_depth: int = 2,
    top_k: int = 8,
    history: Sequence[ChatMessage] = (),
    on_step: Callable[[ResearchStep], None] | None = None,
) -> ResearchAnswer:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if queries_per_question < 1:
        raise ValueError("queries_per_question must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

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
    plan = model.plan_research(
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
    pending_queries = _unique_queries(plan.queries)[:queries_per_question]
    seen_queries: set[str] = set()
    layer_query_count = queries_per_question

    for layer_index in range(max_depth):
        LOGGER.info(
            "research_layer layer=%d queries=%d objectives=%d",
            layer_index + 1,
            len(pending_queries),
            len(plan.objectives),
        )
        for query in pending_queries:
            query_key = query.query.casefold()
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            evidence = _search(store, query, top_k)
            for item in evidence:
                state.evidence_by_ref.setdefault(item.source_ref, item)
            state.searches.append(
                ResearchSearch(
                    query=query,
                    source_refs=[item.source_ref for item in evidence],
                )
            )
            _record_step(
                state,
                ResearchStep(
                    stage="tool",
                    message=(
                        f"claim_search returned {len(evidence)} evidence chunks "
                        f"for {query.query!r}."
                    ),
                    tool_name="claim_search",
                    query=query.query,
                    source_refs=[item.source_ref for item in evidence],
                ),
                on_step,
            )
            LOGGER.info(
                "research_evidence layer=%d query=%r count=%d source_refs=%s",
                layer_index + 1,
                query.query,
                len(evidence),
                [item.source_ref for item in evidence],
            )
            query_findings = model.extract_findings(query, evidence)
            _validate_finding_sources(query_findings, evidence)
            state.findings.extend(query_findings)

        state.findings = _deduplicate_findings(state.findings)
        _record_step(
            state,
            ResearchStep(
                stage="validation",
                message=f"Validated {len(state.findings)} supported findings.",
            ),
            on_step,
        )
        if layer_index == max_depth - 1:
            break

        layer_query_count = max(1, layer_query_count // 2)
        review = model.review_gaps(
            question,
            conversation,
            plan,
            state.findings,
            store.documents,
            layer_query_count,
        )
        state.gap_reviews.append(review)
        _record_step(
            state,
            ResearchStep(
                stage="gap_review",
                message=(
                    "Evidence coverage is complete."
                    if review.complete
                    else (
                        f"Found {len(review.missing_information)} evidence gaps "
                        f"and planned {len(review.queries)} follow-up searches."
                    )
                ),
            ),
            on_step,
        )
        LOGGER.info(
            "research_gap_review layer=%d complete=%s gaps=%d queries=%d",
            layer_index + 1,
            review.complete,
            len(review.missing_information),
            len(review.queries),
        )
        if review.complete:
            break
        pending_queries = [
            query
            for query in _unique_queries(review.queries)
            if query.query.casefold() not in seen_queries
        ][:layer_query_count]
        if not pending_queries:
            break

    draft = model.write_answer(
        question,
        conversation,
        plan,
        state.findings,
    )
    answer_text, answer_sources = _validate_draft(
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
        plan=plan,
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


def _search(store: ClaimStore, query: ResearchQuery, top_k: int) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            document_id=item.document_id,
            document_type=item.document_type,
            document_title=item.document_title,
            page_ids=item.page_ids,
            source_ref=item.source_ref,
            text=item.text,
        )
        for item in store.search(query.query, top_k=top_k)
    ]


def _validate_finding_sources(
    findings: list[ResearchFinding],
    evidence: list[EvidenceItem],
) -> None:
    evidence_refs = {item.source_ref for item in evidence}
    for finding in findings:
        normalized_refs = [
            _normalize_source_ref(source_ref, evidence_refs)
            for source_ref in finding.source_refs
        ]
        invalid_refs = [
            source_ref
            for source_ref in normalized_refs
            if source_ref not in evidence_refs
        ]
        if invalid_refs:
            raise ValueError(
                "Finding cites sources outside retrieved evidence: "
                f"{invalid_refs}"
            )
        finding.source_refs = _unique_text(normalized_refs)


def _validate_draft(
    answer: str,
    declared_sources: list[str],
    findings: list[ResearchFinding],
) -> tuple[str, list[str]]:
    allowed_sources = {
        source_ref
        for finding in findings
        for source_ref in finding.source_refs
    }
    declared_sources = [
        _normalize_source_ref(source_ref, allowed_sources)
        for source_ref in declared_sources
    ]
    invalid_sources = [
        source_ref
        for source_ref in declared_sources
        if source_ref not in allowed_sources
    ]
    if invalid_sources:
        raise ValueError(f"Answer cites sources outside validated findings: {invalid_sources}")
    answer = SOURCE_CITATION.sub(
        lambda match: (
            f"[{_normalize_source_ref(match.group(1), allowed_sources)}]"
        ),
        answer,
    )
    answer_sources = _unique_text(SOURCE_CITATION.findall(answer))
    undeclared_sources = [
        source_ref
        for source_ref in answer_sources
        if source_ref not in declared_sources
    ]
    if undeclared_sources:
        raise ValueError(
            f"Answer contains undeclared source references: {undeclared_sources}"
        )
    if findings and not answer_sources:
        raise ValueError("Answer with factual findings must cite at least one source")
    return answer, answer_sources


def _normalize_source_ref(source_ref: str, allowed_sources: set[str]) -> str:
    if source_ref in allowed_sources:
        return source_ref
    without_page = re.sub(r":p\d+$", "", source_ref)
    return without_page if without_page in allowed_sources else source_ref


def _deduplicate_findings(
    findings: list[ResearchFinding],
) -> list[ResearchFinding]:
    unique: list[ResearchFinding] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for finding in findings:
        key = (finding.insight, tuple(finding.source_refs))
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def _unique_queries(queries: Iterable[ResearchQuery]) -> list[ResearchQuery]:
    unique: list[ResearchQuery] = []
    seen: set[str] = set()
    for query in queries:
        key = query.query.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def _unique_text(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
