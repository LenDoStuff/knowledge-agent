"""Provider-neutral LLM configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, cast

from dotenv import load_dotenv


RuntimeMode = Literal["home", "work"]
LlmProvider = Literal["openrouter", "azure"]
ReasoningEffort = Literal["low", "medium", "high"]


class ConfigurationError(Exception):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class LlmSettings:
    mode: RuntimeMode
    model: str
    reasoning_effort: ReasoningEffort
    openrouter_api_key: str | None = field(repr=False)
    azure_ai_project_endpoint: str | None

    @classmethod
    def from_env(cls) -> "LlmSettings":
        load_dotenv()
        mode = _required_env("KNOWLEDGE_AGENT_MODE").lower()
        if mode not in {"home", "work"}:
            raise ConfigurationError(
                "KNOWLEDGE_AGENT_MODE must be either 'home' or 'work'"
            )

        reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "medium").strip().lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ConfigurationError(
                "LLM_REASONING_EFFORT must be low, medium, or high"
            )

        if mode == "home":
            model = _required_env("OPENROUTER_MODEL")
            openrouter_api_key = _empty_to_none(os.getenv("OPENROUTER_API_KEY"))
            if openrouter_api_key is None:
                raise ConfigurationError(
                    "OPENROUTER_API_KEY is required in home mode"
                )
            azure_endpoint = None
        else:
            model = _required_env("AZURE_OPENAI_MODEL")
            openrouter_api_key = None
            azure_endpoint = _empty_to_none(os.getenv("AZURE_AI_PROJECT_ENDPOINT"))
            if azure_endpoint is None:
                raise ConfigurationError(
                    "AZURE_AI_PROJECT_ENDPOINT is required in work mode"
                )

        return cls(
            mode=cast(RuntimeMode, mode),
            model=model,
            reasoning_effort=cast(ReasoningEffort, reasoning_effort),
            openrouter_api_key=openrouter_api_key,
            azure_ai_project_endpoint=azure_endpoint,
        )

    @property
    def provider(self) -> LlmProvider:
        return "openrouter" if self.mode == "home" else "azure"

    def safe_summary(self) -> dict[str, str | int | None]:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }


def _required_env(name: str) -> str:
    value = _empty_to_none(os.getenv(name))
    if value is None:
        raise ConfigurationError(f"{name} is required")
    return value


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
