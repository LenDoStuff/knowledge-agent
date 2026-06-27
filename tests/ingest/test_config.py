from pathlib import Path

import pytest

from ingest.config import (
    ClaimKbSettings,
    validate_document_intelligence_endpoint,
)
from ingest.exceptions import ConfigurationError


def settings(**updates):
    values = {
        "data_root": Path("data/claims"),
        "document_intelligence_endpoint": None,
        "snowflake_connection_name": "default",
        "snowflake_embedding_model": "snowflake-arctic-embed-l-v2.0",
    }
    values.update(updates)
    return ClaimKbSettings(**values)


def test_home_settings_require_endpoint_and_api_key():
    configured = settings(
        document_intelligence_endpoint=(
            "https://westus.api.cognitive.microsoft.com"
        ),
        document_intelligence_api_key="secret-test-key",
    )

    configured.require_ingestion_settings("home")
    assert "secret-test-key" not in repr(configured)

    with pytest.raises(ConfigurationError, match="API_KEY"):
        settings(
            document_intelligence_endpoint="https://example.cognitiveservices.azure.com"
        ).require_ingestion_settings("home")


def test_work_settings_require_project_connection_and_snowflake():
    configured = settings(
        document_intelligence_connection_name="document-intelligence",
    )

    configured.require_ingestion_settings("work")

    with pytest.raises(ConfigurationError, match="CONNECTION_NAME"):
        settings().require_ingestion_settings("work")

    with pytest.raises(ConfigurationError, match="SNOWFLAKE_CONNECTION_NAME"):
        settings(
            document_intelligence_connection_name="document-intelligence",
            snowflake_connection_name="",
        ).require_ingestion_settings("work")


def test_work_document_intelligence_endpoint_rejects_regional_endpoint():
    with pytest.raises(ConfigurationError, match="custom subdomain"):
        validate_document_intelligence_endpoint(
            "https://westus.api.cognitive.microsoft.com",
            require_custom_subdomain=True,
        )


def test_settings_load_mode_specific_document_intelligence_values(monkeypatch):
    monkeypatch.setattr("ingest.config.load_dotenv", lambda: None)
    for name in [
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

    configured = ClaimKbSettings.from_env()

    assert configured.document_intelligence_api_key == "secret-test-key"
    assert configured.document_intelligence_connection_name == "document-intelligence"
    assert configured.snowflake_connection_name == "default"
    assert "secret-test-key" not in repr(configured)
