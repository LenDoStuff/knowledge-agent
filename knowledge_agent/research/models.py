"""Structured inputs, state summaries, and outputs for claim research."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: NonEmptyText


class ResearchQuery(BaseModel):
    query: NonEmptyText
    research_goal: NonEmptyText


class ResearchPlan(BaseModel):
    objectives: list[NonEmptyText] = Field(min_length=1)
    queries: list[ResearchQuery] = Field(min_length=1)


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


class GapReview(BaseModel):
    complete: bool
    missing_information: list[NonEmptyText] = Field(default_factory=list)
    queries: list[ResearchQuery] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_review_has_no_queries(self) -> "GapReview":
        if self.complete and self.queries:
            raise ValueError("a complete gap review cannot include more queries")
        return self


class ResearchSearch(BaseModel):
    query: ResearchQuery
    source_refs: list[NonEmptyText] = Field(default_factory=list)


class ResearchStep(BaseModel):
    stage: Literal["plan", "tool", "gap_review", "validation", "answer"]
    message: NonEmptyText
    tool_name: str | None = None
    query: str | None = None
    source_refs: list[NonEmptyText] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    answer: NonEmptyText
    source_refs: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("source_refs", mode="before")
    @classmethod
    def remove_citation_brackets(cls, values: object) -> object:
        if not isinstance(values, list):
            return values
        return [
            value.strip()[1:-1]
            if (
                isinstance(value, str)
                and value.strip().startswith("[")
                and value.strip().endswith("]")
            )
            else value
            for value in values
        ]


class ResearchAnswer(BaseModel):
    question: NonEmptyText
    answer: NonEmptyText
    plan: ResearchPlan
    searches: list[ResearchSearch]
    gap_reviews: list[GapReview] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)
    findings: list[ResearchFinding]
    source_refs: list[NonEmptyText]

    @model_validator(mode="after")
    def require_sources_for_findings(self) -> "ResearchAnswer":
        if self.findings and not self.source_refs:
            raise ValueError("an answer with factual findings requires source_refs")
        return self
