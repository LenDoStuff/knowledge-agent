"""LLM boundary for planning, evidence extraction, and answer writing."""

from __future__ import annotations

import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel

from ingest import DocumentMetadata
from infrastructure.responses import StructuredOutputClient
from research.schemas import (
    EvidenceItem,
    NonEmptyText,
    ResearchFinding,
    ResearchQuery,
)


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


class ResearchLlm(Protocol):
    def plan_queries(
        self,
        question: str,
        documents: list[DocumentMetadata],
        breadth: int,
    ) -> list[ResearchQuery]:
        ...

    def extract_findings(
        self,
        query: ResearchQuery,
        evidence: list[EvidenceItem],
    ) -> list[ResearchFinding]:
        ...

    def write_answer(
        self,
        question: str,
        findings: list[ResearchFinding],
    ) -> str:
        ...


class QueryPlan(BaseModel):
    queries: list[ResearchQuery]


class FindingSet(BaseModel):
    findings: list[ResearchFinding]


class WrittenAnswer(BaseModel):
    answer: NonEmptyText


class ResearchResponsesLlm:
    def __init__(self, client: StructuredOutputClient) -> None:
        self._client = client

    def plan_queries(
        self,
        question: str,
        documents: list[DocumentMetadata],
        breadth: int,
    ) -> list[ResearchQuery]:
        document_context = "\n".join(
            f"- {document.id} | {document.document_type} | "
            f"{document.title} | {document.summary}"
            for document in documents
        )
        system = (
                "You plan focused research over one insurance claim knowledge "
                "base. Use only the available document metadata."
            )
        user = (
                f"Research question: {question}\n"
                f"Create up to {breadth} distinct lexical search queries. Each "
                "query should target evidence needed to answer the question.\n\n"
                f"Available documents:\n{document_context}"
            )
        plan = self._parse(
            operation="plan_queries",
            system=system,
            user=user,
            response_model=QueryPlan,
        )
        return plan.queries

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
                "Extract relevant findings. Each finding must include one or "
                "more exact source references that support it. Add focused "
                "follow-up questions only when more claim evidence is needed.\n\n"
                f"Evidence:\n{evidence_context}"
            )
        result = self._parse(
            operation="extract_findings",
            system=system,
            user=user,
            response_model=FindingSet,
        )
        return result.findings

    def write_answer(
        self,
        question: str,
        findings: list[ResearchFinding],
    ) -> str:
        finding_context = "\n".join(
            f"- {finding.insight} "
            f"[{', '.join(finding.source_refs)}]"
            for finding in findings
        )
        system = (
                "You answer a question about one insurance claim. Use only the "
                "provided validated findings. Every factual statement must cite "
                "one or more supporting source references in square brackets."
            )
        user = (
                f"Question: {question}\n\n"
                f"Validated findings:\n{finding_context}\n\n"
                "Write a concise answer. If there are no findings, state that "
                "the claim knowledge base does not contain enough evidence."
            )
        result = self._parse(
            operation="write_answer",
            system=system,
            user=user,
            response_model=WrittenAnswer,
        )
        return result.answer

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
