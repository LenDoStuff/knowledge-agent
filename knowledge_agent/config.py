"""Application-level environment configuration."""

from __future__ import annotations

import os
from typing import Literal, cast

from dotenv import load_dotenv


DeploymentProfile = Literal["api_key", "azure_project", "snowflake"]
DEFAULT_SNOWFLAKE_CONNECTION_NAME = "default"


class ConfigurationError(Exception):
    """Raised when required application configuration is missing or invalid."""


def load_profile() -> DeploymentProfile:
    load_dotenv()
    value = required_env("KNOWLEDGE_AGENT_PROFILE").lower()
    if value not in {"api_key", "azure_project", "snowflake"}:
        raise ConfigurationError(
            "KNOWLEDGE_AGENT_PROFILE must be 'api_key', 'azure_project', "
            "or 'snowflake'"
        )
    return cast(DeploymentProfile, value)


def required_env(name: str) -> str:
    value = optional_env(name)
    if value is None:
        raise ConfigurationError(f"{name} is required")
    return value


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
