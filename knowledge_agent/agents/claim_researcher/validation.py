"""Source and citation validation for claim research."""

from __future__ import annotations

import re
from collections.abc import Iterable

from knowledge_agent.agents.claim_researcher.models import (
    EvidenceItem,
    ResearchFinding,
    ResearchQuery,
)


SOURCE_CITATION = re.compile(r"\[([^\[\]\s]+/[^\[\]\s]+#[^\[\]\s]+)\]")


def validate_finding_sources(
    findings: list[ResearchFinding],
    evidence: list[EvidenceItem],
) -> None:
    evidence_refs = {item.source_ref for item in evidence}
    for finding in findings:
        invalid_refs = [
            source_ref
            for source_ref in finding.source_refs
            if source_ref not in evidence_refs
        ]
        if invalid_refs:
            raise ValueError(
                "Finding cites sources outside retrieved evidence: "
                f"{invalid_refs}"
            )
        finding.source_refs = unique_text(finding.source_refs)


def validate_draft(
    answer: str,
    declared_sources: list[str],
    findings: list[ResearchFinding],
) -> tuple[str, list[str]]:
    allowed_sources = {
        source_ref
        for finding in findings
        for source_ref in finding.source_refs
    }
    invalid_sources = [
        source_ref
        for source_ref in declared_sources
        if source_ref not in allowed_sources
    ]
    if invalid_sources:
        raise ValueError(f"Answer cites sources outside validated findings: {invalid_sources}")
    answer_sources = unique_text(SOURCE_CITATION.findall(answer))
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


def deduplicate_findings(
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


def unique_queries(queries: Iterable[ResearchQuery]) -> list[ResearchQuery]:
    unique: list[ResearchQuery] = []
    seen: set[str] = set()
    for query in queries:
        key = query.query.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def unique_text(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
