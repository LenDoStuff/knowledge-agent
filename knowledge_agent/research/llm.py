"""LLM boundary for research planning, gap review, and grounded writing."""

from __future__ import annotations

import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel

from knowledge_agent.claims.models import DocumentMetadata
from knowledge_agent.llm.client import StructuredOutputClient
from knowledge_agent.research.models import (
    ChatMessage,
    DraftAnswer,
    EvidenceItem,
    GapReview,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
)


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


class ResearchModel(Protocol):
    def plan_research(
        self,
        question: str,
        history: list[ChatMessage],
        documents: list[DocumentMetadata],
        queries_per_question: int,
    ) -> ResearchPlan:
        ...

    def extract_findings(
        self,
        query: ResearchQuery,
        evidence: list[EvidenceItem],
    ) -> list[ResearchFinding]:
        ...

    def review_gaps(
        self,
        question: str,
        history: list[ChatMessage],
        plan: ResearchPlan,
        findings: list[ResearchFinding],
        documents: list[DocumentMetadata],
        query_limit: int,
    ) -> GapReview:
        ...

    def write_answer(
        self,
        question: str,
        history: list[ChatMessage],
        plan: ResearchPlan,
        findings: list[ResearchFinding],
    ) -> DraftAnswer:
        ...


class FindingSet(BaseModel):
    findings: list[ResearchFinding]


class ResponsesResearchModel:
    def __init__(self, client: StructuredOutputClient) -> None:
        self._client = client

    def plan_research(
        self,
        question: str,
        history: list[ChatMessage],
        documents: list[DocumentMetadata],
        queries_per_question: int,
    ) -> ResearchPlan:
        system = (
            "You plan focused research over one insurance claim knowledge "
            "base. Resolve conversational references from the prior history, "
            "then define explicit answer objectives and lexical search queries. "
            "Use only the available document metadata."
        )
        user = (
            f"Prior conversation:\n{_history_context(history)}\n\n"
            f"Current question: {question}\n"
            f"Create up to {queries_per_question} distinct search queries and "
            "the concrete objectives a supported answer must satisfy.\n\n"
            f"Available documents:\n{_document_context(documents)}"
        )
        return self._parse(
            operation="plan_research",
            system=system,
            user=user,
            response_model=ResearchPlan,
        )

    def extract_findings(
        self,
        query: ResearchQuery,
        evidence: list[EvidenceItem],
    ) -> list[ResearchFinding]:
        evidence_context = "\n\n".join(
            f"Source: {item.source_ref}\n"
            f"Document: {item.document_title} ({item.document_type})\n"
            f"Pages: {', '.join(item.page_ids)}\n"
            f"Text:\n{item.text}"
            for item in evidence
        )
        system = (
            "You extract factual findings from claim evidence. Use only the "
            "provided evidence and cite only its exact source references."
        )
        user = (
            f"Search query: {query.query}\n"
            f"Research goal: {query.research_goal}\n\n"
            "Extract only relevant, supported findings. Each finding must "
            "include one or more exact source references. Copy each Source "
            "value verbatim; do not append page IDs such as :p5 because Pages "
            "are separate metadata. Return no findings when the evidence is "
            "insufficient.\n\n"
            f"Evidence:\n{evidence_context or '(no evidence retrieved)'}"
        )
        result = self._parse(
            operation="extract_findings",
            system=system,
            user=user,
            response_model=FindingSet,
        )
        return result.findings

    def review_gaps(
        self,
        question: str,
        history: list[ChatMessage],
        plan: ResearchPlan,
        findings: list[ResearchFinding],
        documents: list[DocumentMetadata],
        query_limit: int,
    ) -> GapReview:
        system = (
            "You review research coverage over one insurance claim. Compare "
            "the validated findings with every plan objective. Mark the work "
            "complete when the evidence is sufficient; otherwise create only "
            "focused lexical queries for material gaps."
        )
        user = (
            f"Prior conversation:\n{_history_context(history)}\n\n"
            f"Current question: {question}\n"
            f"Objectives:\n{_list_context(plan.objectives)}\n\n"
            f"Validated findings:\n{_finding_context(findings)}\n\n"
            f"Available documents:\n{_document_context(documents)}\n\n"
            f"Return at most {query_limit} new queries. Do not repeat a query "
            "unless it targets genuinely missing information."
        )
        return self._parse(
            operation="review_gaps",
            system=system,
            user=user,
            response_model=GapReview,
        )

    def write_answer(
        self,
        question: str,
        history: list[ChatMessage],
        plan: ResearchPlan,
        findings: list[ResearchFinding],
    ) -> DraftAnswer:
        system = (
            "You answer a question about one insurance claim. Use only the "
            "validated findings. Every factual statement must cite an exact "
            "source reference in its own square brackets. Declare precisely "
            "the source references that appear in the answer."
        )
        user = (
            f"Prior conversation:\n{_history_context(history)}\n\n"
            f"Current question: {question}\n"
            f"Answer objectives:\n{_list_context(plan.objectives)}\n\n"
            f"Validated findings:\n{_finding_context(findings)}\n\n"
            "Write a concise conversational answer. If there are no findings, "
            "state that the claim knowledge base does not contain enough "
            "evidence and return an empty source_refs list. In source_refs, "
            "return each raw source reference without square brackets; use "
            "square brackets only for citations in the answer text."
        )
        return self._parse(
            operation="write_answer",
            system=system,
            user=user,
            response_model=DraftAnswer,
        )

    def _parse(
        self,
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
        parsed = self._client.parse(system, user, response_model)
        LOGGER.debug(
            "research_llm_output operation=%s response_model=%s output=%s",
            operation,
            response_model.__name__,
            parsed.model_dump_json(),
        )
        return parsed


def _history_context(history: list[ChatMessage]) -> str:
    if not history:
        return "(no prior conversation)"
    return "\n".join(f"{message.role}: {message.content}" for message in history)


def _document_context(documents: list[DocumentMetadata]) -> str:
    return "\n".join(
        f"- {document.id} | {document.document_type} | "
        f"{document.title} | {document.summary}"
        for document in documents
    )


def _finding_context(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "(no validated findings)"
    return "\n".join(
        f"- {finding.insight} [{', '.join(finding.source_refs)}]"
        for finding in findings
    )


def _list_context(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)
