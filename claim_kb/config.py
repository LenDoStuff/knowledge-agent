"""Runtime configuration for claim knowledge base operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from claim_kb.exceptions import ConfigurationError


DEFAULT_SNOWFLAKE_CONNECTION_NAME = "default"
DEFAULT_SNOWFLAKE_EMBEDDING_MODEL = "snowflake-arctic-embed-l-v2.0"


@dataclass(frozen=True)
class ClaimKbSettings:
    data_root: Path
    ai_project_endpoint: str | None
    document_intelligence_endpoint: str | None
    openai_deployment: str | None
    tenant_id: str | None
    snowflake_connection_name: str
    snowflake_embedding_model: str

    @classmethod
    def from_env(cls) -> "ClaimKbSettings":
        return cls(
            data_root=Path(os.getenv("CLAIM_KB_DATA_ROOT", "data/claims")),
            ai_project_endpoint=_empty_to_none(os.getenv("AZURE_AI_PROJECT_ENDPOINT")),
            document_intelligence_endpoint=_empty_to_none(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            ),
            openai_deployment=_empty_to_none(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
            tenant_id=_empty_to_none(os.getenv("AZURE_TENANT_ID")),
            snowflake_connection_name=(
                _empty_to_none(os.getenv("SNOWFLAKE_CONNECTION_NAME"))
                or DEFAULT_SNOWFLAKE_CONNECTION_NAME
            ),
            snowflake_embedding_model=(
                _empty_to_none(os.getenv("SNOWFLAKE_EMBEDDING_MODEL"))
                or DEFAULT_SNOWFLAKE_EMBEDDING_MODEL
            ),
        )

    def require_ingestion_settings(self) -> None:
        missing = [
            name
            for name, value in [
                ("AZURE_AI_PROJECT_ENDPOINT", self.ai_project_endpoint),
                (
                    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
                    self.document_intelligence_endpoint,
                ),
                ("AZURE_OPENAI_DEPLOYMENT", self.openai_deployment),
            ]
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        self.validate_document_intelligence_endpoint()

    def require_retrieval_settings(self) -> None:
        if not self.snowflake_connection_name:
            raise ConfigurationError("SNOWFLAKE_CONNECTION_NAME cannot be empty")
        if not self.snowflake_embedding_model:
            raise ConfigurationError("SNOWFLAKE_EMBEDDING_MODEL cannot be empty")

    def validate_document_intelligence_endpoint(self) -> None:
        endpoint = self.document_intelligence_endpoint
        if not endpoint:
            raise ConfigurationError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required")
        parsed = urlparse(endpoint)
        host = parsed.netloc.lower()
        if not parsed.scheme or not host:
            raise ConfigurationError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT must be an absolute URL"
            )
        if host.endswith(".api.cognitive.microsoft.com") or ".api.cognitive." in host:
            raise ConfigurationError(
                "Document Intelligence Microsoft Entra auth requires a custom "
                "subdomain endpoint, not a regional endpoint"
            )


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
