"""LLM provider configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, cast

from knowledge_agent.config import (
    ConfigurationError,
    DEFAULT_SNOWFLAKE_CONNECTION_NAME,
    DeploymentProfile,
    optional_env,
    required_env,
)


LlmProvider = Literal["nvidia", "azure", "snowflake"]
ReasoningEffort = Literal["low", "medium", "high"]
NVIDIA_DEEPSEEK_MODEL = "deepseek-ai/deepseek-v4-pro"


@dataclass(frozen=True)
class LlmSettings:
    profile: DeploymentProfile
    model: str
    reasoning_effort: ReasoningEffort
    nvidia_base_url: str | None = None
    nvidia_api_key_ds4: str | None = field(default=None, repr=False)
    azure_ai_project_endpoint: str | None = None
    snowflake_connection_name: str | None = None
    snowflake_cortex_pat: str | None = field(default=None, repr=False)


def load_llm_settings(profile: DeploymentProfile) -> LlmSettings:
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "medium").strip().lower()
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ConfigurationError("LLM_REASONING_EFFORT must be low, medium, or high")

    if profile == "api_key":
        return LlmSettings(
            profile=profile,
            model=NVIDIA_DEEPSEEK_MODEL,
            reasoning_effort=cast(ReasoningEffort, reasoning_effort),
            nvidia_base_url=required_env("nvidia_base_url"),
            nvidia_api_key_ds4=required_env("nvidia_api_key_ds4"),
        )

    if profile == "azure_project":
        return LlmSettings(
            profile=profile,
            model=required_env("AZURE_OPENAI_MODEL"),
            reasoning_effort=cast(ReasoningEffort, reasoning_effort),
            azure_ai_project_endpoint=required_env("AZURE_AI_PROJECT_ENDPOINT"),
        )

    return LlmSettings(
        profile=profile,
        model=required_env("SNOWFLAKE_CORTEX_MODEL"),
        reasoning_effort=cast(ReasoningEffort, reasoning_effort),
        snowflake_connection_name=(
            optional_env("SNOWFLAKE_CONNECTION_NAME")
            or DEFAULT_SNOWFLAKE_CONNECTION_NAME
        ),
        snowflake_cortex_pat=required_env("SNOWFLAKE_CORTEX_PAT"),
    )


def llm_provider(settings: LlmSettings) -> LlmProvider:
    if settings.profile == "api_key":
        return "nvidia"
    if settings.profile == "azure_project":
        return "azure"
    return "snowflake"
