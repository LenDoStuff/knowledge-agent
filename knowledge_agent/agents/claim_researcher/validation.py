"""Citation validation for claim research output."""

from __future__ import annotations

import re
from collections.abc import Iterable

from knowledge_agent.agents.claim_researcher.models import ClaimResearchOutput


SOURCE_CITATION = re.compile(r"\[([^\[\]\s]+/[^\[\]\s]+#[^\[\]\s]+)\]")


def validate_research_output(
    output: ClaimResearchOutput,
    retrieved_source_refs: set[str],
) -> ClaimResearchOutput:
    answer_sources = unique_text(SOURCE_CITATION.findall(output.answer))
    if output.source_refs != answer_sources:
        raise ValueError(
            "source_refs must exactly match inline citations in first-appearance order"
        )
    invalid_sources = [
        source_ref
        for source_ref in output.source_refs
        if source_ref not in retrieved_source_refs
    ]
    if invalid_sources:
        raise ValueError(
            f"Answer cites sources outside retrieved evidence: {invalid_sources}"
        )
    if output.evidence_sufficient and not output.source_refs:
        raise ValueError("An evidence-sufficient answer requires at least one citation")
    return output


def unique_text(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
