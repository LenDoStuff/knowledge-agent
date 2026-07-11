"""Structured inputs, state summaries, and outputs for claim research."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
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


class ResearchLlmAuditEntry(BaseModel):
    kind: Literal["llm"] = "llm"
    operation: Literal[
        "plan_research",
        "extract_findings",
        "review_gaps",
        "write_answer",
    ]
    response_model: NonEmptyText
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    result: dict[str, object] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_result_or_error(self) -> "ResearchLlmAuditEntry":
        if (self.result is None) == (self.error is None):
            raise ValueError("an LLM audit entry requires exactly one result or error")
        return self


class ResearchToolAuditEntry(BaseModel):
    kind: Literal["tool"] = "tool"
    tool_name: Literal["claim_search"] = "claim_search"
    query: ResearchQuery
    top_k: int = Field(ge=1)
    result: list[EvidenceItem] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_result_or_error(self) -> "ResearchToolAuditEntry":
        if (self.result is None) == (self.error is None):
            raise ValueError("a tool audit entry requires exactly one result or error")
        return self


ResearchAuditEntry = Annotated[
    ResearchLlmAuditEntry | ResearchToolAuditEntry,
    Field(discriminator="kind"),
]


class ResearchAuditTrail(BaseModel):
    entries: list[ResearchAuditEntry] = Field(default_factory=list)


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
