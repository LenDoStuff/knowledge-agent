"""Structured inputs and outputs for claim research."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ResearchQuery(BaseModel):
    query: NonEmptyText
    research_goal: NonEmptyText


class EvidenceItem(BaseModel):
    document_id: str
    document_type: str
    document_title: str
    page_ids: list[str]
    source_ref: NonEmptyText
    text: str


class ResearchFinding(BaseModel):
    insight: NonEmptyText
    source_refs: list[NonEmptyText] = Field(min_length=1)
    follow_up_questions: list[NonEmptyText] = Field(default_factory=list)


class ResearchAnswer(BaseModel):
    question: NonEmptyText
    answer: str
    findings: list[ResearchFinding]
    source_refs: list[NonEmptyText]

    @model_validator(mode="after")
    def require_sources_for_findings(self) -> "ResearchAnswer":
        if self.findings and not self.source_refs:
            raise ValueError("an answer with factual findings requires source_refs")
        return self
