from pathlib import Path

import pytest

from claim_kb.config import ClaimKbSettings
from claim_kb.exceptions import ConfigurationError


def test_document_intelligence_endpoint_rejects_regional_endpoint():
    settings = ClaimKbSettings(
        data_root=Path("data/claims"),
        ai_project_endpoint="https://example.services.ai.azure.com/api/projects/proj",
        document_intelligence_endpoint="https://westus.api.cognitive.microsoft.com",
        openai_deployment="gpt-test",
        tenant_id=None,
        snowflake_connection_name="default",
        snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
    )

    with pytest.raises(ConfigurationError):
        settings.validate_document_intelligence_endpoint()


def test_settings_do_not_require_api_key_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ignored")
    monkeypatch.setenv("DOCUMENTINTELLIGENCE_API_KEY", "ignored")
    monkeypatch.setenv(
        "AZURE_AI_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/proj",
    )
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "https://example.cognitiveservices.azure.com",
    )
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-test")

    settings = ClaimKbSettings.from_env()
    settings.require_ingestion_settings()

    assert settings.ai_project_endpoint is not None
    assert settings.openai_deployment == "gpt-test"
    assert settings.snowflake_connection_name == "default"
    assert settings.snowflake_embedding_model == "snowflake-arctic-embed-l-v2.0"
    assert not hasattr(settings, "openai_api_key")
    assert not hasattr(settings, "document_intelligence_api_key")
    assert not hasattr(settings, "embedding_deployment")
