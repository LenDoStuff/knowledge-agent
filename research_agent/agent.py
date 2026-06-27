"""Fixed-depth research loop over one persisted claim knowledge base."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from claim_kb import ClaimKbKnowledgeStore
from research_agent.llm import ResearchLlm
from research_agent.schemas import EvidenceItem, ResearchAnswer, ResearchFinding


LOGGER = logging.getLogger(__name__)


def run_claim_research(
    claim_path: str | Path,
    question: str,
    llm: ResearchLlm,
    breadth: int = 4,
    depth: int = 2,
    top_k: int = 8,
) -> ResearchAnswer:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if breadth < 1:
        raise ValueError("breadth must be at least 1")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    LOGGER.info(
        "research_start question=%r breadth=%d depth=%d top_k=%d",
        question,
        breadth,
        depth,
        top_k,
    )
    store = ClaimKbKnowledgeStore(claim_path)
    layer_questions = [question]
    layer_breadth = breadth
    findings: list[ResearchFinding] = []

    for layer_index in range(depth):
        LOGGER.info(
            "research_layer layer=%d breadth=%d questions=%d",
            layer_index + 1,
            layer_breadth,
            len(layer_questions),
        )
        layer_findings: list[ResearchFinding] = []
        for layer_question in layer_questions:
            queries = llm.plan_queries(
                layer_question,
                store.manifest.documents,
                layer_breadth,
            )
            LOGGER.info(
                "research_queries layer=%d question=%r queries=%s",
                layer_index + 1,
                layer_question,
                [query.query for query in queries],
            )
            for query in queries:
                evidence = [
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
                LOGGER.info(
                    "research_evidence layer=%d query=%r count=%d source_refs=%s",
                    layer_index + 1,
                    query.query,
                    len(evidence),
                    [item.source_ref for item in evidence],
                )
                evidence_refs = {item.source_ref for item in evidence}
                query_findings = llm.extract_findings(query, evidence)
                for finding in query_findings:
                    invalid_refs = [
                        source_ref
                        for source_ref in finding.source_refs
                        if source_ref not in evidence_refs
                    ]
                    if invalid_refs:
                        raise ValueError(
                            f"Finding cites sources outside retrieved evidence: "
                            f"{invalid_refs}"
                        )
                    findings.append(finding)
                    layer_findings.append(finding)
                LOGGER.info(
                    "research_findings layer=%d query=%r accepted=%d "
                    "source_refs=%s follow_ups=%d",
                    layer_index + 1,
                    query.query,
                    len(query_findings),
                    _unique_text(
                        source_ref
                        for finding in query_findings
                        for source_ref in finding.source_refs
                    ),
                    sum(
                        len(finding.follow_up_questions)
                        for finding in query_findings
                    ),
                )

        if layer_index == depth - 1:
            break
        layer_questions = _unique_text(
            question
            for finding in layer_findings
            for question in finding.follow_up_questions
        )
        if not layer_questions:
            break
        LOGGER.info(
            "research_followups next_layer=%d questions=%s",
            layer_index + 2,
            layer_questions,
        )
        layer_breadth = max(1, layer_breadth // 2)

    findings = _deduplicate_findings(findings)
    source_refs = _unique_text(
        source_ref
        for finding in findings
        for source_ref in finding.source_refs
    )
    LOGGER.info(
        "research_answer findings=%d source_refs=%s",
        len(findings),
        source_refs,
    )
    answer_text = llm.write_answer(question, findings)
    LOGGER.info(
        "research_complete findings=%d sources=%d",
        len(findings),
        len(source_refs),
    )
    return ResearchAnswer(
        question=question,
        answer=answer_text,
        findings=findings,
        source_refs=source_refs,
    )


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


def _unique_text(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
