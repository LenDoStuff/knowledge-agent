"""Structured claim-search evidence and final research output."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvidenceItem(BaseModel):
    document_id: str
    document_type: str
    document_title: str
    page_ids: list[str]
    source_ref: NonEmptyText
    text: str


class ResearchClarification(BaseModel):
    question: NonEmptyText
    reason: NonEmptyText


class ResearchPlanStep(BaseModel):
    query: NonEmptyText
    research_goal: NonEmptyText


class ClaimResearchPlan(BaseModel):
    objective: NonEmptyText
    understood_scope: NonEmptyText
    assumptions: list[NonEmptyText] = Field(default_factory=list)
    searches: list[ResearchPlanStep] = Field(min_length=1)
    completion_criteria: list[NonEmptyText] = Field(min_length=1)


class ClaimResearchOutput(BaseModel):
    answer: NonEmptyText
    source_refs: list[NonEmptyText] = Field(default_factory=list)
    evidence_sufficient: bool

    @model_validator(mode="after")
    def insufficient_output_has_no_sources(self) -> "ClaimResearchOutput":
        if not self.evidence_sufficient and self.source_refs:
            raise ValueError("insufficient-evidence output cannot declare sources")
        return self
