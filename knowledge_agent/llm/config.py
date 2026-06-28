"""LLM provider configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, cast

from knowledge_agent.config import (
    ConfigurationError,
    DeploymentProfile,
    required_env,
)


LlmProvider = Literal["openrouter", "azure"]
ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class LlmSettings:
    profile: DeploymentProfile
    model: str
    reasoning_effort: ReasoningEffort
    openrouter_api_key: str | None = field(default=None, repr=False)
    azure_ai_project_endpoint: str | None = None

    @classmethod
    def from_env(cls, profile: DeploymentProfile) -> "LlmSettings":
        reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "medium").strip().lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ConfigurationError(
                "LLM_REASONING_EFFORT must be low, medium, or high"
            )

        if profile == "api_key":
            return cls(
                profile=profile,
                model=required_env("OPENROUTER_MODEL"),
                reasoning_effort=cast(ReasoningEffort, reasoning_effort),
                openrouter_api_key=required_env("OPENROUTER_API_KEY"),
            )

        return cls(
            profile=profile,
            model=required_env("AZURE_OPENAI_MODEL"),
            reasoning_effort=cast(ReasoningEffort, reasoning_effort),
            azure_ai_project_endpoint=required_env("AZURE_AI_PROJECT_ENDPOINT"),
        )

    @property
    def provider(self) -> LlmProvider:
        return "openrouter" if self.profile == "api_key" else "azure"
