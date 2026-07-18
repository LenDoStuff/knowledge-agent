"""Configuration for claim ingestion and semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from urllib.parse import urlparse

from knowledge_agent.config import (
    ConfigurationError,
    DEFAULT_SNOWFLAKE_CONNECTION_NAME,
    DeploymentProfile,
    optional_env,
)


DEFAULT_SNOWFLAKE_EMBEDDING_MODEL = "snowflake-arctic-embed-l-v2.0"
DEFAULT_SNOWFLAKE_DOCUMENT_STAGE = "KNOWLEDGE_AGENT_DOCUMENTS"
SNOWFLAKE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class ClaimSettings:
    data_root: Path
    document_intelligence_endpoint: str | None
    snowflake_connection_name: str
    snowflake_embedding_model: str
    snowflake_document_stage: str = DEFAULT_SNOWFLAKE_DOCUMENT_STAGE
    document_intelligence_api_key: str | None = field(default=None, repr=False)
    document_intelligence_connection_name: str | None = None


def load_claim_settings() -> ClaimSettings:
    return ClaimSettings(
        data_root=Path(optional_env("CLAIM_DATA_ROOT") or "data/claims"),
        document_intelligence_endpoint=optional_env(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
        ),
        document_intelligence_api_key=optional_env(
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
        ),
        document_intelligence_connection_name=optional_env(
            "AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME"
        ),
        snowflake_connection_name=(
            optional_env("SNOWFLAKE_CONNECTION_NAME")
            or DEFAULT_SNOWFLAKE_CONNECTION_NAME
        ),
        snowflake_embedding_model=(
            optional_env("SNOWFLAKE_EMBEDDING_MODEL")
            or DEFAULT_SNOWFLAKE_EMBEDDING_MODEL
        ),
        snowflake_document_stage=(
            optional_env("SNOWFLAKE_DOCUMENT_STAGE")
            or DEFAULT_SNOWFLAKE_DOCUMENT_STAGE
        ),
    )


def require_ingestion_settings(
    settings: ClaimSettings,
    profile: DeploymentProfile,
) -> None:
    if profile == "api_key":
        if not settings.document_intelligence_endpoint:
            raise ConfigurationError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required for "
                "the api_key profile"
            )
        if not settings.document_intelligence_api_key:
            raise ConfigurationError(
                "AZURE_DOCUMENT_INTELLIGENCE_API_KEY is required for "
                "the api_key profile"
            )
        validate_document_intelligence_endpoint(
            settings.document_intelligence_endpoint,
            require_custom_subdomain=False,
        )
        return

    if profile == "snowflake":
        require_semantic_retrieval_settings(settings)
        validate_snowflake_stage_name(settings.snowflake_document_stage)
        return

    if not settings.document_intelligence_connection_name:
        raise ConfigurationError(
            "AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME is required for "
            "the azure_project profile"
        )
    require_semantic_retrieval_settings(settings)


def require_semantic_retrieval_settings(settings: ClaimSettings) -> None:
    if not settings.snowflake_connection_name:
        raise ConfigurationError("SNOWFLAKE_CONNECTION_NAME cannot be empty")
    if not settings.snowflake_embedding_model:
        raise ConfigurationError("SNOWFLAKE_EMBEDDING_MODEL cannot be empty")


def validate_snowflake_stage_name(stage_name: str) -> None:
    if not SNOWFLAKE_IDENTIFIER_PATTERN.fullmatch(stage_name):
        raise ConfigurationError(
            "SNOWFLAKE_DOCUMENT_STAGE must be one unquoted Snowflake identifier"
        )


def validate_document_intelligence_endpoint(
    endpoint: str,
    *,
    require_custom_subdomain: bool,
) -> None:
    parsed = urlparse(endpoint)
    host = parsed.netloc.lower()
    if not parsed.scheme or not host:
        raise ConfigurationError(
            "Document Intelligence endpoint must be an absolute URL"
        )
    if require_custom_subdomain and (
        host.endswith(".api.cognitive.microsoft.com")
        or ".api.cognitive." in host
    ):
        raise ConfigurationError(
            "Document Intelligence Microsoft Entra auth requires a custom "
            "subdomain endpoint, not a regional endpoint"
        )
