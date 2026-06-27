"""Runtime configuration for claim knowledge base operations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse

from dotenv import load_dotenv

from knowledge_agent.infrastructure.config import ConfigurationError, RuntimeMode


DEFAULT_SNOWFLAKE_CONNECTION_NAME = "default"
DEFAULT_SNOWFLAKE_EMBEDDING_MODEL = "snowflake-arctic-embed-l-v2.0"
EmbeddingMode = Literal["snowflake", "none"]


def validate_embedding_mode(value: str) -> EmbeddingMode:
    if value not in {"snowflake", "none"}:
        raise ValueError("embedding_mode must be 'snowflake' or 'none'")
    return cast(EmbeddingMode, value)


@dataclass(frozen=True)
class ClaimKbSettings:
    data_root: Path
    document_intelligence_endpoint: str | None
    snowflake_connection_name: str
    snowflake_embedding_model: str
    document_intelligence_api_key: str | None = field(default=None, repr=False)
    document_intelligence_connection_name: str | None = None

    @classmethod
    def from_env(cls) -> "ClaimKbSettings":
        load_dotenv()
        return cls(
            data_root=Path(os.getenv("CLAIM_KB_DATA_ROOT", "data/claims")),
            document_intelligence_endpoint=_empty_to_none(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            ),
            document_intelligence_api_key=_empty_to_none(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
            ),
            document_intelligence_connection_name=_empty_to_none(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME")
            ),
            snowflake_connection_name=(
                _empty_to_none(os.getenv("SNOWFLAKE_CONNECTION_NAME"))
                or DEFAULT_SNOWFLAKE_CONNECTION_NAME
            ),
            snowflake_embedding_model=(
                _empty_to_none(os.getenv("SNOWFLAKE_EMBEDDING_MODEL"))
                or DEFAULT_SNOWFLAKE_EMBEDDING_MODEL
            ),
        )

    def require_ingestion_settings(self, mode: RuntimeMode) -> None:
        if mode == "home":
            if not self.document_intelligence_endpoint:
                raise ConfigurationError(
                    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required in home mode"
                )
            if not self.document_intelligence_api_key:
                raise ConfigurationError(
                    "AZURE_DOCUMENT_INTELLIGENCE_API_KEY is required in home mode"
                )
            validate_document_intelligence_endpoint(
                self.document_intelligence_endpoint,
                require_custom_subdomain=False,
            )
            return

        if mode == "work":
            if not self.document_intelligence_connection_name:
                raise ConfigurationError(
                    "AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME is required "
                    "in work mode"
                )
            self.require_retrieval_settings()
            return

        raise ConfigurationError(f"Unsupported runtime mode: {mode}")

    def require_retrieval_settings(self) -> None:
        if not self.snowflake_connection_name:
            raise ConfigurationError("SNOWFLAKE_CONNECTION_NAME cannot be empty")
        if not self.snowflake_embedding_model:
            raise ConfigurationError("SNOWFLAKE_EMBEDDING_MODEL cannot be empty")

    def validate_document_intelligence_endpoint(self) -> None:
        endpoint = self.document_intelligence_endpoint
        if not endpoint:
            raise ConfigurationError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required")
        validate_document_intelligence_endpoint(
            endpoint,
            require_custom_subdomain=False,
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


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
