"""LLM functions for research planning, gap review, and grounded writing."""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from knowledge_agent.agents.claim_researcher.models import (
    ChatMessage,
    DraftAnswer,
    EvidenceItem,
    GapReview,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
)
from knowledge_agent.agents.claim_researcher.prompts import (
    build_extract_findings_prompt,
    build_plan_research_prompt,
    build_review_gaps_prompt,
    build_write_answer_prompt,
)
from knowledge_agent.claims.models import DocumentMetadata
from knowledge_agent.llm.client import StructuredOutputParser


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


class FindingSet(BaseModel):
    findings: list[ResearchFinding]


def plan_research(
    parse_structured_output: StructuredOutputParser,
    question: str,
    history: list[ChatMessage],
    documents: list[DocumentMetadata],
    queries_per_question: int,
) -> ResearchPlan:
    system, user = build_plan_research_prompt(
        question,
        history,
        documents,
        queries_per_question,
    )
    return _parse_research_output(
        parse_structured_output,
        operation="plan_research",
        system=system,
        user=user,
        response_model=ResearchPlan,
    )


def extract_findings(
    parse_structured_output: StructuredOutputParser,
    query: ResearchQuery,
    evidence: list[EvidenceItem],
) -> list[ResearchFinding]:
    system, user = build_extract_findings_prompt(query, evidence)
    result = _parse_research_output(
        parse_structured_output,
        operation="extract_findings",
        system=system,
        user=user,
        response_model=FindingSet,
    )
    return result.findings


def review_gaps(
    parse_structured_output: StructuredOutputParser,
    question: str,
    history: list[ChatMessage],
    plan: ResearchPlan,
    findings: list[ResearchFinding],
    documents: list[DocumentMetadata],
    query_limit: int,
) -> GapReview:
    system, user = build_review_gaps_prompt(
        question,
        history,
        plan,
        findings,
        documents,
        query_limit,
    )
    return _parse_research_output(
        parse_structured_output,
        operation="review_gaps",
        system=system,
        user=user,
        response_model=GapReview,
    )


def write_answer(
    parse_structured_output: StructuredOutputParser,
    question: str,
    history: list[ChatMessage],
    plan: ResearchPlan,
    findings: list[ResearchFinding],
) -> DraftAnswer:
    system, user = build_write_answer_prompt(question, history, plan, findings)
    return _parse_research_output(
        parse_structured_output,
        operation="write_answer",
        system=system,
        user=user,
        response_model=DraftAnswer,
    )


def _parse_research_output(
    parse_structured_output: StructuredOutputParser,
    *,
    operation: str,
    system: str,
    user: str,
    response_model: type[ParsedModel],
) -> ParsedModel:
    LOGGER.debug(
        "research_llm_prompt operation=%s response_model=%s "
        "system=%r user=%r",
        operation,
        response_model.__name__,
        system,
        user,
    )
    parsed = parse_structured_output(system, user, response_model)
    LOGGER.debug(
        "research_llm_output operation=%s response_model=%s output=%s",
        operation,
        response_model.__name__,
        parsed.model_dump_json(),
    )
    return parsed
