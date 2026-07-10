"""Live dependency construction for claims."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import Iterator, cast

from azure.core.credentials import AzureKeyCredential

from knowledge_agent.agents.document_classifier import (
    classify_document,
    classify_page_boundary,
    extract_document_metadata,
)
from knowledge_agent.claims.config import (
    ClaimSettings,
    require_ingestion_settings,
    require_semantic_retrieval_settings,
    validate_document_intelligence_endpoint,
)
from knowledge_agent.claims.embeddings import SnowflakeAiEmbedder
from knowledge_agent.claims.filesystem import read_json
from knowledge_agent.claims.models import ClaimManifest
from knowledge_agent.claims.ocr import AzureDocumentIntelligenceOcrClient
from knowledge_agent.claims.pipeline import IngestionServices
from knowledge_agent.claims.store import ClaimStore, load_claim_store
from knowledge_agent.claims.vector_store import ChromaVectorStore
from knowledge_agent.config import ConfigurationError
from knowledge_agent.llm.client import parse_structured_output
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import open_provider_clients


@contextmanager
def live_ingestion_services(
    claim_id: str,
    settings: ClaimSettings,
    llm_settings: LlmSettings,
) -> Iterator[IngestionServices]:
    require_ingestion_settings(settings, llm_settings.profile)
    with open_provider_clients(llm_settings) as provider, ExitStack() as resources:
        parse = partial(parse_structured_output, llm_settings, provider.openai)
        embedder = None
        vector_store_factory = None

        if llm_settings.profile == "api_key":
            endpoint = cast(str, settings.document_intelligence_endpoint)
            credential = AzureKeyCredential(
                cast(str, settings.document_intelligence_api_key)
            )
            retrieval_mode = "lexical"
        else:
            if provider.azure_project is None or provider.azure_credential is None:
                raise ConfigurationError(
                    "azure_project profile requires Azure project clients"
                )
            connection = provider.azure_project.connections.get(
                cast(str, settings.document_intelligence_connection_name),
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
            credential = provider.azure_credential
            retrieval_mode = "semantic"
            embedder = SnowflakeAiEmbedder(
                settings.snowflake_connection_name,
                settings.snowflake_embedding_model,
            )
            resources.callback(embedder.close)
            vector_store_factory = lambda root: ChromaVectorStore(
                claim_id,
                root / "index" / "chroma",
            )

        ocr_client = AzureDocumentIntelligenceOcrClient(endpoint, credential)
        resources.callback(ocr_client.close)
        yield IngestionServices(
            ocr_client=ocr_client,
            classify_document=partial(classify_document, parse),
            classify_page_boundary=partial(classify_page_boundary, parse),
            extract_document_metadata=partial(extract_document_metadata, parse),
            embedder=embedder,
            vector_store_factory=vector_store_factory,
            retrieval_mode=retrieval_mode,
        )


@contextmanager
def open_claim_store(
    claim_path: str | Path,
    settings: ClaimSettings,
) -> Iterator[ClaimStore]:
    claim_path = Path(claim_path)
    manifest_data = read_json(claim_path / "manifest.json")
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest.json must contain a JSON object")
    manifest = ClaimManifest.model_validate(manifest_data)
    if manifest.retrieval_mode == "lexical":
        yield load_claim_store(claim_path)
        return

    require_semantic_retrieval_settings(settings)
    if manifest.embedding_provider != "snowflake" or not manifest.embedding_model:
        raise ConfigurationError(
            "Only Snowflake embeddings are supported for semantic claims"
        )
    with ExitStack() as resources:
        embedder = SnowflakeAiEmbedder(
            settings.snowflake_connection_name,
            manifest.embedding_model,
        )
        resources.callback(embedder.close)
        vector_store = ChromaVectorStore(
            manifest.claim_id,
            claim_path / "index" / "chroma",
        )
        resources.callback(vector_store.close)
        yield load_claim_store(
            claim_path,
            embedder=embedder,
            vector_store=vector_store,
        )
