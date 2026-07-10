from pathlib import Path

import pytest

from knowledge_agent.claims.config import (
    ClaimSettings,
    load_claim_settings,
    require_ingestion_settings,
    validate_document_intelligence_endpoint,
)
from knowledge_agent.config import ConfigurationError


def settings(**updates):
    values = {
        "data_root": Path("data/claims"),
        "document_intelligence_endpoint": None,
        "snowflake_connection_name": "default",
        "snowflake_embedding_model": "snowflake-arctic-embed-l-v2.0",
    }
    values.update(updates)
    return ClaimSettings(**values)


def test_api_key_profile_requires_endpoint_and_api_key():
    configured = settings(
        document_intelligence_endpoint=(
            "https://westus.api.cognitive.microsoft.com"
        ),
        document_intelligence_api_key="secret-test-key",
    )

    require_ingestion_settings(configured, "api_key")
    assert "secret-test-key" not in repr(configured)

    with pytest.raises(ConfigurationError, match="API_KEY"):
        require_ingestion_settings(
            settings(
                document_intelligence_endpoint=(
                    "https://example.cognitiveservices.azure.com"
                )
            ),
            "api_key",
        )


def test_azure_project_profile_requires_connection_and_snowflake():
    configured = settings(
        document_intelligence_connection_name="document-intelligence",
    )

    require_ingestion_settings(configured, "azure_project")

    with pytest.raises(ConfigurationError, match="CONNECTION_NAME"):
        require_ingestion_settings(settings(), "azure_project")

    with pytest.raises(ConfigurationError, match="SNOWFLAKE_CONNECTION_NAME"):
        require_ingestion_settings(
            settings(
                document_intelligence_connection_name="document-intelligence",
                snowflake_connection_name="",
            ),
            "azure_project",
        )


def test_azure_project_endpoint_rejects_regional_endpoint():
    with pytest.raises(ConfigurationError, match="custom subdomain"):
        validate_document_intelligence_endpoint(
            "https://westus.api.cognitive.microsoft.com",
            require_custom_subdomain=True,
        )


def test_settings_load_mode_specific_document_intelligence_values(monkeypatch):
    for name in [
        "CLAIM_DATA_ROOT",
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
        "AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME",
        "SNOWFLAKE_CONNECTION_NAME",
        "SNOWFLAKE_EMBEDDING_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "https://example.cognitiveservices.azure.com",
    )
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", "secret-test-key")
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME",
        "document-intelligence",
    )

    configured = load_claim_settings()

    assert configured.document_intelligence_api_key == "secret-test-key"
    assert configured.document_intelligence_connection_name == "document-intelligence"
    assert configured.snowflake_connection_name == "default"
    assert configured.data_root == Path("data/claims")
    assert "secret-test-key" not in repr(configured)
