"""Internal workflow state for claim research."""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_agent.agents.claim_researcher.models import (
    EvidenceItem,
    GapReview,
    ResearchAuditEntry,
    ResearchFinding,
    ResearchPlan,
    ResearchQuery,
    ResearchSearch,
    ResearchStep,
)


@dataclass
class ResearchState:
    plan: ResearchPlan
    searches: list[ResearchSearch] = field(default_factory=list)
    gap_reviews: list[GapReview] = field(default_factory=list)
    steps: list[ResearchStep] = field(default_factory=list)
    evidence_by_ref: dict[str, EvidenceItem] = field(default_factory=dict)
    findings: list[ResearchFinding] = field(default_factory=list)


@dataclass(frozen=True)
class _QueryResult:
    query: ResearchQuery
    evidence: list[EvidenceItem]
    findings: list[ResearchFinding]
    audit_entries: list[ResearchAuditEntry] = field(default_factory=list)
    error: Exception | None = None
