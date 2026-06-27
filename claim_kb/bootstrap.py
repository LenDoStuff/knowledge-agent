"""Live dependency construction for Claim KB operations."""

from __future__ import annotations

from typing import cast

from azure.core.credentials import AzureKeyCredential

from claim_kb.classify import ResponsesClaimClassifier
from claim_kb.config import (
    ClaimKbSettings,
    validate_document_intelligence_endpoint,
)
from claim_kb.embeddings import SnowflakeAiEmbedder
from claim_kb.filesystem import claim_root
from claim_kb.ingest import IngestionServices
from claim_kb.ocr import AzureDocumentIntelligenceOcrClient
from claim_kb.vector_store import ChromaVectorStore
from knowledge_agent.infrastructure import (
    ConfigurationError,
    LlmSettings,
    PortableResponsesClient,
    create_openai_runtime,
)


def build_live_ingestion_services(
    claim_id: str,
    settings: ClaimKbSettings,
) -> IngestionServices:
    llm_settings = LlmSettings.from_env()
    settings.require_ingestion_settings(llm_settings.mode)
    runtime = create_openai_runtime(llm_settings)
    responses = PortableResponsesClient(
        llm_settings,
        runtime,
    )
    resources = [responses]
    try:
        if llm_settings.mode == "home":
            endpoint = cast(str, settings.document_intelligence_endpoint)
            api_key = cast(str, settings.document_intelligence_api_key)
            ocr_credential = AzureKeyCredential(api_key)
            embedding_mode = "none"
        else:
            project = runtime.project_client
            ocr_credential = runtime.azure_credential
            if project is None or ocr_credential is None:
                raise ConfigurationError(
                    "Work mode requires an Azure AI Projects runtime"
                )
            connection_name = cast(
                str,
                settings.document_intelligence_connection_name,
            )
            connection = project.connections.get(
                connection_name,
                include_credentials=False,
            )
            endpoint = str(getattr(connection, "target", "")).strip()
            if not endpoint:
                raise ConfigurationError(
                    "Document Intelligence project connection has no target endpoint"
                )
            validate_document_intelligence_endpoint(
                endpoint,
                require_custom_subdomain=True,
            )
            embedding_mode = "snowflake"

        ocr_client = AzureDocumentIntelligenceOcrClient(
            endpoint,
            ocr_credential,
        )
        resources.append(ocr_client)
        embedder = None
        vector_store_factory = None
        if llm_settings.mode == "work":
            embedder = SnowflakeAiEmbedder(
                settings.snowflake_connection_name,
                settings.snowflake_embedding_model,
            )
            resources.append(embedder)
            vector_store_factory = lambda root: ChromaVectorStore(
                claim_id,
                root / "index" / "chroma",
            )
        return IngestionServices(
            ocr_client=ocr_client,
            classifier=ResponsesClaimClassifier(responses),
            embedder=embedder,
            vector_store_factory=vector_store_factory,
            embedding_mode=embedding_mode,
            close=lambda: _close_resources(resources),
        )
    except Exception:
        _close_resources(resources)
        raise


def build_live_embedder(settings: ClaimKbSettings) -> SnowflakeAiEmbedder:
    return SnowflakeAiEmbedder(
        settings.snowflake_connection_name,
        settings.snowflake_embedding_model,
    )


def build_live_vector_store(
    settings: ClaimKbSettings,
    claim_id: str,
) -> ChromaVectorStore:
    return ChromaVectorStore(
        claim_id,
        claim_root(settings.data_root, claim_id) / "index" / "chroma",
    )


def _close_resources(resources: list[object]) -> None:
    first_error: Exception | None = None
    for resource in reversed(resources):
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error
