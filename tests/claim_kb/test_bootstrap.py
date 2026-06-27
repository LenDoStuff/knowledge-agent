from pathlib import Path
from types import SimpleNamespace

import pytest
from azure.core.credentials import AzureKeyCredential

from claim_kb.bootstrap import build_live_ingestion_services
from claim_kb.config import ClaimKbSettings
from claim_kb.exceptions import ConfigurationError
from knowledge_agent.infrastructure.config import LlmSettings


class FakeResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRuntime(FakeResource):
    def __init__(self, project_client=None, azure_credential=None) -> None:
        super().__init__()
        self.client = object()
        self.project_client = project_client
        self.azure_credential = azure_credential


class FakeEmbedder(FakeResource):
    embedding_provider = "snowflake"
    embedding_model = "test-model"


def home_llm_settings() -> LlmSettings:
    return LlmSettings(
        mode="home",
        model="provider/model",
        reasoning_effort="medium",
        openrouter_api_key="secret-openrouter-key",
        azure_ai_project_endpoint=None,
    )


def work_llm_settings() -> LlmSettings:
    return LlmSettings(
        mode="work",
        model="deployment",
        reasoning_effort="medium",
        openrouter_api_key=None,
        azure_ai_project_endpoint=(
            "https://example.services.ai.azure.com/api/projects/project"
        ),
    )


def claim_settings(**updates) -> ClaimKbSettings:
    values = {
        "data_root": Path("data/claims"),
        "document_intelligence_endpoint": None,
        "snowflake_connection_name": "default",
        "snowflake_embedding_model": "snowflake-model",
    }
    values.update(updates)
    return ClaimKbSettings(**values)


def set_llm_settings(monkeypatch, settings):
    monkeypatch.setattr(
        "claim_kb.bootstrap.LlmSettings.from_env",
        classmethod(lambda cls: settings),
    )


def test_home_builds_key_ocr_without_snowflake_or_chroma(monkeypatch):
    runtime = FakeRuntime()
    ocr = FakeResource()
    ocr_calls = []
    set_llm_settings(monkeypatch, home_llm_settings())
    monkeypatch.setattr(
        "claim_kb.bootstrap.create_openai_runtime",
        lambda settings: runtime,
    )
    monkeypatch.setattr(
        "claim_kb.bootstrap.AzureDocumentIntelligenceOcrClient",
        lambda endpoint, credential: ocr_calls.append((endpoint, credential)) or ocr,
    )
    monkeypatch.setattr(
        "claim_kb.bootstrap.SnowflakeAiEmbedder",
        lambda *args: pytest.fail("home mode must not construct Snowflake"),
    )

    services = build_live_ingestion_services(
        "CLM-HOME",
        claim_settings(
            document_intelligence_endpoint=(
                "https://example.cognitiveservices.azure.com"
            ),
            document_intelligence_api_key="secret-document-key",
        ),
    )

    assert services.embedding_mode == "none"
    assert services.embedder is None
    assert services.vector_store_factory is None
    assert ocr_calls[0][0] == "https://example.cognitiveservices.azure.com"
    assert isinstance(ocr_calls[0][1], AzureKeyCredential)

    services.close()
    assert ocr.closed
    assert runtime.closed


def test_work_resolves_project_ocr_and_builds_snowflake(monkeypatch):
    connection_calls = []
    connections = SimpleNamespace(
        get=lambda name, include_credentials: connection_calls.append(
            (name, include_credentials)
        )
        or SimpleNamespace(
            target="https://documents.cognitiveservices.azure.com"
        )
    )
    project = SimpleNamespace(connections=connections)
    credential = FakeResource()
    runtime = FakeRuntime(project, credential)
    ocr = FakeResource()
    embedder = FakeEmbedder()
    ocr_calls = []
    snowflake_calls = []
    set_llm_settings(monkeypatch, work_llm_settings())
    monkeypatch.setattr(
        "claim_kb.bootstrap.create_openai_runtime",
        lambda settings: runtime,
    )
    monkeypatch.setattr(
        "claim_kb.bootstrap.AzureDocumentIntelligenceOcrClient",
        lambda endpoint, passed_credential: ocr_calls.append(
            (endpoint, passed_credential)
        )
        or ocr,
    )
    monkeypatch.setattr(
        "claim_kb.bootstrap.SnowflakeAiEmbedder",
        lambda connection_name, model: snowflake_calls.append(
            (connection_name, model)
        )
        or embedder,
    )

    services = build_live_ingestion_services(
        "CLM-WORK",
        claim_settings(
            document_intelligence_connection_name="document-intelligence",
        ),
    )

    assert connection_calls == [("document-intelligence", False)]
    assert ocr_calls == [
        ("https://documents.cognitiveservices.azure.com", credential)
    ]
    assert snowflake_calls == [("default", "snowflake-model")]
    assert services.embedding_mode == "snowflake"
    assert services.embedder is embedder
    assert callable(services.vector_store_factory)

    services.close()
    assert embedder.closed
    assert ocr.closed
    assert runtime.closed


def test_work_rejects_project_connection_without_target(monkeypatch):
    project = SimpleNamespace(
        connections=SimpleNamespace(
            get=lambda name, include_credentials: SimpleNamespace(target="")
        )
    )
    runtime = FakeRuntime(project, FakeResource())
    set_llm_settings(monkeypatch, work_llm_settings())
    monkeypatch.setattr(
        "claim_kb.bootstrap.create_openai_runtime",
        lambda settings: runtime,
    )

    with pytest.raises(ConfigurationError, match="no target endpoint"):
        build_live_ingestion_services(
            "CLM-WORK",
            claim_settings(
                document_intelligence_connection_name="document-intelligence",
            ),
        )

    assert runtime.closed
