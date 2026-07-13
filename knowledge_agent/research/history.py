"""Typed, atomic persistence for auditable claim-research interactions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter

from knowledge_agent.agents.claim_researcher import (
    ClaimResearchOutput,
    ClaimResearchPlan,
)
from knowledge_agent.claims.models import RetrievalMode


HISTORY_VERSION = 1
ACTIVE_STATUSES = {
    "planning",
    "awaiting_clarification",
    "awaiting_approval",
    "researching",
}
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}
InteractionStatus = Literal[
    "planning",
    "awaiting_clarification",
    "awaiting_approval",
    "researching",
    "completed",
    "cancelled",
    "failed",
]
_HISTORY_LOCK = RLock()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InteractionUsage(BaseModel):
    requests: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    def plus(self, other: "InteractionUsage") -> "InteractionUsage":
        return InteractionUsage(
            requests=self.requests + other.requests,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ClarificationExchange(BaseModel):
    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    answer: str | None = None


class KnowledgeBaseSnapshot(BaseModel):
    retrieval_mode: RetrievalMode
    embedding_provider: str | None = None
    embedding_model: str | None = None
    lightrag_version: str | None = None
    lightrag_index_claim_id: str | None = None
    lightrag_index_llm_provider: str | None = None
    lightrag_index_llm_model: str | None = None
    lightrag_embedding_dimension: int | None = Field(default=None, gt=0)
    lightrag_embedding_max_tokens: int | None = Field(default=None, gt=0)
    lightrag_query_mode: Literal["hybrid"] | None = None
    lightrag_indexed_chunk_count: int | None = Field(default=None, ge=0)
    lightrag_entity_count: int | None = Field(default=None, ge=0)
    lightrag_relationship_count: int | None = Field(default=None, ge=0)
    lightrag_indexing_usage: InteractionUsage | None = None
    lightrag_index_created_at: datetime | None = None


class ResearchInteraction(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    claim_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: InteractionStatus
    question: str = Field(min_length=1)
    planning_enabled: bool
    knowledge_base: KnowledgeBaseSnapshot | None = None
    clarifications: list[ClarificationExchange] = Field(default_factory=list)
    plan: ClaimResearchPlan | None = None
    output: ClaimResearchOutput | None = None
    agent_messages: list[ModelMessage] = Field(default_factory=list)
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    planning_usage: InteractionUsage = Field(default_factory=InteractionUsage)
    research_usage: InteractionUsage = Field(default_factory=InteractionUsage)
    error: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "ResearchInteraction":
        ModelMessagesTypeAdapter.validate_python(self.agent_messages)
        if self.status == "awaiting_approval" and self.plan is None:
            raise ValueError("awaiting_approval interaction requires a plan")
        if self.status == "completed" and self.output is None:
            raise ValueError("completed interaction requires an output")
        if self.status == "failed" and not self.error:
            raise ValueError("failed interaction requires an error")
        return self

    @property
    def usage(self) -> InteractionUsage:
        return self.planning_usage.plus(self.research_usage)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES


class ResearchHistory(BaseModel):
    version: Literal[1] = HISTORY_VERSION
    claim_id: str = Field(min_length=1)
    interactions: list[ResearchInteraction] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_interactions(self) -> "ResearchHistory":
        ids = [interaction.id for interaction in self.interactions]
        if len(ids) != len(set(ids)):
            raise ValueError("research history contains duplicate interaction IDs")
        if any(item.claim_id != self.claim_id for item in self.interactions):
            raise ValueError("research interaction claim_id does not match history")
        if sum(item.is_active for item in self.interactions) > 1:
            raise ValueError("research history cannot contain multiple active interactions")
        return self

    def active_interaction(self) -> ResearchInteraction | None:
        return next((item for item in self.interactions if item.is_active), None)


def research_history_path(claim_path: str | Path) -> Path:
    return Path(claim_path) / "research" / "history.json"


def load_research_history(
    claim_path: str | Path,
    expected_claim_id: str,
) -> ResearchHistory:
    path = research_history_path(claim_path)
    if not path.exists():
        return ResearchHistory(claim_id=expected_claim_id)
    history = ResearchHistory.model_validate_json(path.read_text(encoding="utf-8"))
    if history.claim_id != expected_claim_id:
        raise ValueError(
            f"research history claim_id {history.claim_id!r} does not match "
            f"{expected_claim_id!r}"
        )
    return history


def save_research_history(claim_path: str | Path, history: ResearchHistory) -> None:
    path = research_history_path(claim_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    payload = json.dumps(history.model_dump(mode="json"), indent=2)
    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(path)


def store_interaction(
    claim_path: str | Path,
    interaction: ResearchInteraction,
) -> ResearchHistory:
    with _HISTORY_LOCK:
        history = load_research_history(claim_path, interaction.claim_id)
        existing_index = next(
            (
                index
                for index, item in enumerate(history.interactions)
                if item.id == interaction.id
            ),
            None,
        )
        if existing_index is None:
            if interaction.is_active and history.active_interaction() is not None:
                raise ValueError("this claim already has an active research interaction")
            history.interactions.append(interaction)
        else:
            history.interactions[existing_index] = interaction
        history = ResearchHistory.model_validate(history.model_dump())
        save_research_history(claim_path, history)
        return history


def delete_interaction(
    claim_path: str | Path,
    claim_id: str,
    interaction_id: str,
) -> ResearchHistory:
    with _HISTORY_LOCK:
        history = load_research_history(claim_path, claim_id)
        item = next(
            (item for item in history.interactions if item.id == interaction_id),
            None,
        )
        if item is None:
            raise KeyError(f"research interaction not found: {interaction_id}")
        if item.is_active:
            raise ValueError("active research interactions cannot be deleted")
        history.interactions = [
            item for item in history.interactions if item.id != interaction_id
        ]
        save_research_history(claim_path, history)
        return history


def clear_terminal_interactions(
    claim_path: str | Path,
    claim_id: str,
) -> ResearchHistory:
    with _HISTORY_LOCK:
        history = load_research_history(claim_path, claim_id)
        history.interactions = [
            item for item in history.interactions if item.status not in TERMINAL_STATUSES
        ]
        save_research_history(claim_path, history)
        return history
