from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from azure.core.credentials import AzureKeyCredential

from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import live_ingestion_services, open_claim_store
from knowledge_agent.config import ConfigurationError
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import ProviderClients
from knowledge_agent.claims.vector_store import VectorSearchHit


class FakeResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEmbedder(FakeResource):
    embedding_provider = "snowflake"
    embedding_model = "test-model"


def api_key_llm_settings() -> LlmSettings:
    return LlmSettings(
        profile="api_key",
        model="provider/model",
        reasoning_effort="medium",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_api_key_ds4="secret-nvidia-key",
    )


def azure_llm_settings() -> LlmSettings:
    return LlmSettings(
        profile="azure_project",
        model="deployment",
        reasoning_effort="medium",
        azure_ai_project_endpoint="https://project.example",
    )


def claim_settings(**updates) -> ClaimSettings:
    values = {
        "data_root": Path("data/claims"),
        "document_intelligence_endpoint": None,
        "snowflake_connection_name": "default",
        "snowflake_embedding_model": "snowflake-model",
    }
    values.update(updates)
    return ClaimSettings(**values)


def provider_context(clients):
    @contextmanager
    def open_clients(settings):
        yield clients

    return open_clients


def test_api_key_profile_builds_key_ocr_without_semantic_dependencies(monkeypatch):
    ocr = FakeResource()
    ocr_calls = []
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.open_provider_clients",
        provider_context(ProviderClients(openai=object())),
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.AzureDocumentIntelligenceOcrClient",
        lambda endpoint, credential: ocr_calls.append((endpoint, credential)) or ocr,
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.SnowflakeAiEmbedder",
        lambda *args: pytest.fail("api_key profile must not construct Snowflake"),
    )

    with live_ingestion_services(
        "CLM-API",
        claim_settings(
            document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
            document_intelligence_api_key="secret-document-key",
        ),
        api_key_llm_settings(),
    ) as services:
        assert services.retrieval_mode == "lexical"
        assert services.embedder is None
        assert services.vector_store_factory is None
        assert isinstance(ocr_calls[0][1], AzureKeyCredential)
        assert not ocr.closed

    assert ocr.closed


def test_azure_project_profile_builds_snowflake_and_chroma(monkeypatch):
    connection_calls = []
    connections = SimpleNamespace(
        get=lambda name, include_credentials: connection_calls.append(
            (name, include_credentials)
        )
        or SimpleNamespace(target="https://documents.cognitiveservices.azure.com")
    )
    project = SimpleNamespace(connections=connections)
    credential = FakeResource()
    ocr = FakeResource()
    embedder = FakeEmbedder()
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.open_provider_clients",
        provider_context(
            ProviderClients(
                openai=object(),
                azure_project=project,
                azure_credential=credential,
            )
        ),
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.AzureDocumentIntelligenceOcrClient",
        lambda endpoint, passed_credential: ocr,
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.SnowflakeAiEmbedder",
        lambda connection_name, model: embedder,
    )

    with live_ingestion_services(
        "CLM-AZURE",
        claim_settings(document_intelligence_connection_name="document-intelligence"),
        azure_llm_settings(),
    ) as services:
        assert services.retrieval_mode == "semantic"
        assert services.embedder is embedder
        assert callable(services.vector_store_factory)

    assert connection_calls == [("document-intelligence", False)]
    assert embedder.closed
    assert ocr.closed


def test_azure_project_rejects_connection_without_target(monkeypatch):
    project = SimpleNamespace(
        connections=SimpleNamespace(
            get=lambda name, include_credentials: SimpleNamespace(target="")
        )
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.open_provider_clients",
        provider_context(
            ProviderClients(
                openai=object(),
                azure_project=project,
                azure_credential=FakeResource(),
            )
        ),
    )
    with pytest.raises(ConfigurationError, match="no target endpoint"):
        with live_ingestion_services(
            "CLM-AZURE",
            claim_settings(
                document_intelligence_connection_name="document-intelligence"
            ),
            azure_llm_settings(),
        ):
            pass


def test_semantic_claim_store_uses_manifest_model_and_closes_resources(
    monkeypatch, tmp_path
):
    source = Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
    claim_path = tmp_path / "claim"
    shutil.copytree(source, claim_path)
    manifest_path = claim_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        retrieval_mode="semantic",
        embedding_provider="snowflake",
        embedding_model="stored-model",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    created = []

    class SearchEmbedder(FakeEmbedder):
        def embed_texts(self, texts):
            return [[1.0]]

    class SearchVector(FakeResource):
        def search(self, query_embedding, document_types, top_k):
            return [VectorSearchHit("DOC-002-CHUNK-001", 0.8)]

    embedder = SearchEmbedder()
    vector = SearchVector()
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.SnowflakeAiEmbedder",
        lambda connection, model: created.append((connection, model)) or embedder,
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.ChromaVectorStore",
        lambda claim_id, path: vector,
    )

    with open_claim_store(claim_path, claim_settings()) as store:
        assert store.search("repair", top_k=1)[0].chunk_id == "DOC-002-CHUNK-001"

    assert created == [("default", "stored-model")]
    assert embedder.closed
    assert vector.closed
