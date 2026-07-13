"""Live dependency construction for claims."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Iterator, cast
from uuid import uuid4

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
from knowledge_agent.claims.filesystem import read_json, write_claim_manifest
from knowledge_agent.claims.lightrag import (
    embedding_spec,
    index_lightrag_chunks,
    open_lightrag_resource,
    validate_lightrag_index,
)
from knowledge_agent.claims.models import ClaimManifest, KnowledgeBaseEngine
from knowledge_agent.claims.ocr import AzureDocumentIntelligenceOcrClient
from knowledge_agent.claims.pipeline import IngestionServices
from knowledge_agent.claims.store import ClaimStore, load_claim_store
from knowledge_agent.claims.vector_store import ChromaVectorStore
from knowledge_agent.config import ConfigurationError
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import open_agent_runtime


@contextmanager
def live_ingestion_services(
    claim_id: str,
    settings: ClaimSettings,
    llm_settings: LlmSettings,
    knowledge_base: KnowledgeBaseEngine = "custom",
) -> Iterator[IngestionServices]:
    require_ingestion_settings(settings, llm_settings.profile)
    with open_agent_runtime(llm_settings) as runtime, ExitStack() as resources:
        embedder = None
        vector_store_factory = None
        lightrag_indexer = None
        selected_spec = None

        if llm_settings.profile == "api_key":
            endpoint = cast(str, settings.document_intelligence_endpoint)
            credential = AzureKeyCredential(
                cast(str, settings.document_intelligence_api_key)
            )
            retrieval_mode = "lexical"
        else:
            if runtime.azure_project is None or runtime.azure_credential is None:
                raise ConfigurationError(
                    "azure_project profile requires Azure project clients"
                )
            connection = runtime.azure_project.connections.get(
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
            credential = runtime.azure_credential
            retrieval_mode = "semantic"
            embedder = SnowflakeAiEmbedder(
                settings.snowflake_connection_name,
                settings.snowflake_embedding_model,
            )
            resources.callback(embedder.close)
            def vector_store_factory(root: Path) -> ChromaVectorStore:
                return ChromaVectorStore(
                    claim_id,
                    root / "index" / "chroma",
                )

        if knowledge_base == "lightrag":
            retrieval_mode = "lightrag"
            selected_spec = embedding_spec(
                llm_settings,
                settings.snowflake_embedding_model,
            )

            def lightrag_indexer(index_path: Path, chunks):
                return index_lightrag_chunks(
                    runtime,
                    llm_settings,
                    index_path,
                    claim_id,
                    chunks,
                    selected_spec,
                    snowflake_embedder=embedder,
                )

        ocr_client = AzureDocumentIntelligenceOcrClient(endpoint, credential)
        resources.callback(ocr_client.close)
        yield IngestionServices(
            ocr_client=ocr_client,
            classify_document=partial(classify_document, runtime),
            classify_page_boundary=partial(classify_page_boundary, runtime),
            extract_document_metadata=partial(extract_document_metadata, runtime),
            embedder=embedder,
            vector_store_factory=vector_store_factory,
            retrieval_mode=retrieval_mode,
            lightrag_indexer=lightrag_indexer,
            embedding_provider=(selected_spec.provider if selected_spec else None),
            embedding_model=(selected_spec.model if selected_spec else None),
        )


@contextmanager
def open_claim_store(
    claim_path: str | Path,
    settings: ClaimSettings,
    *,
    runtime=None,
    llm_settings: LlmSettings | None = None,
) -> Iterator[ClaimStore]:
    claim_path = Path(claim_path)
    manifest_data = read_json(claim_path / "manifest.json")
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest.json must contain a JSON object")
    manifest = ClaimManifest.model_validate(manifest_data)
    if manifest.retrieval_mode == "lexical":
        yield load_claim_store(claim_path)
        return

    if manifest.retrieval_mode == "lightrag":
        if runtime is None or llm_settings is None:
            raise ConfigurationError(
                "LightRAG claims require an AgentRuntime and LLM settings"
            )
        metadata = validate_lightrag_index(
            claim_path / "index" / "lightrag",
            manifest,
        )
        selected_spec = embedding_spec(
            llm_settings,
            settings.snowflake_embedding_model,
        )
        if (
            selected_spec.provider != manifest.embedding_provider
            or selected_spec.model != manifest.embedding_model
            or selected_spec.dimension != metadata.embedding_dimension
            or selected_spec.max_tokens != metadata.embedding_max_tokens
        ):
            raise ConfigurationError(
                "Configured LightRAG embedding does not match the persisted index"
            )
        with ExitStack() as resources:
            embedder = None
            if selected_spec.provider == "snowflake":
                embedder = SnowflakeAiEmbedder(
                    settings.snowflake_connection_name,
                    selected_spec.model,
                )
                resources.callback(embedder.close)
            lightrag = resources.enter_context(
                open_lightrag_resource(
                    runtime,
                    llm_settings,
                    claim_path / "index" / "lightrag",
                    selected_spec,
                    snowflake_embedder=embedder,
                    metadata=metadata,
                )
            )
            yield load_claim_store(claim_path, lightrag=lightrag)
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


def rebuild_claim_knowledge_base(
    claim_path: str | Path,
    target: KnowledgeBaseEngine,
    settings: ClaimSettings,
    llm_settings: LlmSettings,
) -> ClaimManifest:
    """Rebuild only the retrieval index from persisted claim chunks."""

    claim_path = Path(claim_path)
    store = load_claim_store(claim_path)
    staging_root = Path(mkdtemp(prefix=".index-build-", dir=claim_path))
    try:
        if target == "lightrag":
            selected_spec = embedding_spec(
                llm_settings,
                settings.snowflake_embedding_model,
            )
            with open_agent_runtime(llm_settings) as runtime, ExitStack() as resources:
                embedder = None
                if selected_spec.provider == "snowflake":
                    embedder = SnowflakeAiEmbedder(
                        settings.snowflake_connection_name,
                        selected_spec.model,
                    )
                    resources.callback(embedder.close)
                index_lightrag_chunks(
                    runtime,
                    llm_settings,
                    staging_root / "index" / "lightrag",
                    store.manifest.claim_id,
                    store.chunks,
                    selected_spec,
                    snowflake_embedder=embedder,
                )
            manifest = ClaimManifest.model_validate(
                store.manifest.model_dump()
                | {
                    "retrieval_mode": "lightrag",
                    "embedding_provider": selected_spec.provider,
                    "embedding_model": selected_spec.model,
                }
            )
            validate_lightrag_index(
                staging_root / "index" / "lightrag",
                manifest,
            )
        elif llm_settings.profile == "api_key":
            manifest = ClaimManifest.model_validate(
                store.manifest.model_dump()
                | {
                    "retrieval_mode": "lexical",
                    "embedding_provider": None,
                    "embedding_model": None,
                }
            )
        else:
            embedder = SnowflakeAiEmbedder(
                settings.snowflake_connection_name,
                settings.snowflake_embedding_model,
            )
            try:
                embeddings = embedder.embed_texts(
                    [chunk.text for chunk in store.chunks]
                )
            finally:
                embedder.close()
            vector_store = ChromaVectorStore(
                store.manifest.claim_id,
                staging_root / "index" / "chroma",
            )
            try:
                vector_store.index_chunks(store.chunks, embeddings)
            finally:
                vector_store.close()
            manifest = ClaimManifest.model_validate(
                store.manifest.model_dump()
                | {
                    "retrieval_mode": "semantic",
                    "embedding_provider": "snowflake",
                    "embedding_model": settings.snowflake_embedding_model,
                }
            )

        _commit_rebuilt_index(claim_path, staging_root / "index", manifest)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _commit_rebuilt_index(
    claim_path: Path,
    staged_index: Path,
    manifest: ClaimManifest,
) -> None:
    current_index = claim_path / "index"
    backup_index = claim_path / f".index-backup-{uuid4().hex}"
    moved_current = False
    committed = False
    try:
        if current_index.exists():
            current_index.replace(backup_index)
            moved_current = True
        if staged_index.exists():
            staged_index.replace(current_index)
        write_claim_manifest(claim_path, manifest)
        committed = True
    except Exception:
        if current_index.exists():
            shutil.rmtree(current_index)
        if moved_current and backup_index.exists():
            backup_index.replace(current_index)
        raise
    if committed:
        shutil.rmtree(backup_index, ignore_errors=True)
