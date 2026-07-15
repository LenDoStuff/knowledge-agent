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
    LLM_CACHE_FILE,
    embedding_spec,
    index_lightrag_chunks,
    open_lightrag_resource,
    validate_lightrag_index,
)
from knowledge_agent.claims.models import (
    ClaimManifest,
    KnowledgeBaseEngine,
    RetrievalMode,
)
from knowledge_agent.claims.ocr import AzureDocumentIntelligenceOcrClient
from knowledge_agent.claims.pipeline import IngestionServices
from knowledge_agent.claims.store import ClaimStore, load_claim_store
from knowledge_agent.claims.vector_store import ChromaVectorStore
from knowledge_agent.config import ConfigurationError
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import open_agent_runtime


LIGHTRAG_REBUILD_CACHE_FILE = ".lightrag-rebuild-cache.json"


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
            custom_mode: RetrievalMode = "lexical"
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
            custom_mode = "semantic"
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

        retrieval_modes = _retrieval_modes(knowledge_base, custom_mode)
        if "lightrag" in retrieval_modes:
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
            retrieval_mode=retrieval_modes[0],
            additional_retrieval_modes=retrieval_modes[1:],
            lightrag_indexer=lightrag_indexer,
            embedding_provider=(selected_spec.provider if selected_spec else None),
            embedding_model=(selected_spec.model if selected_spec else None),
        )


@contextmanager
def open_claim_store(
    claim_path: str | Path,
    settings: ClaimSettings,
    *,
    retrieval_mode: RetrievalMode | None = None,
    runtime=None,
    llm_settings: LlmSettings | None = None,
) -> Iterator[ClaimStore]:
    claim_path = Path(claim_path)
    manifest_data = read_json(claim_path / "manifest.json")
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest.json must contain a JSON object")
    manifest = ClaimManifest.model_validate(manifest_data)
    selected_mode = retrieval_mode or manifest.retrieval_mode
    if selected_mode not in manifest.available_retrieval_modes:
        raise ValueError(
            f"Retrieval mode {selected_mode!r} is not available for "
            f"claim {manifest.claim_id}"
        )
    if selected_mode == "lexical":
        yield load_claim_store(claim_path, retrieval_mode=selected_mode)
        return

    if selected_mode == "lightrag":
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
            yield load_claim_store(
                claim_path,
                retrieval_mode=selected_mode,
                lightrag=lightrag,
            )
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
            retrieval_mode=selected_mode,
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
    store = load_claim_store(claim_path, validate_index=False)
    staging_root = Path(mkdtemp(prefix=".index-build-", dir=claim_path))
    rebuild_cache_path = claim_path / LIGHTRAG_REBUILD_CACHE_FILE
    try:
        custom_mode: RetrievalMode = (
            "lexical" if llm_settings.profile == "api_key" else "semantic"
        )
        retrieval_modes = _retrieval_modes(target, custom_mode)
        selected_spec = None
        if "lightrag" in retrieval_modes:
            selected_spec = embedding_spec(
                llm_settings,
                settings.snowflake_embedding_model,
            )

        with ExitStack() as resources:
            embedder = None
            if "semantic" in retrieval_modes or (
                selected_spec is not None and selected_spec.provider == "snowflake"
            ):
                require_semantic_retrieval_settings(settings)
                embedding_model = (
                    selected_spec.model
                    if selected_spec is not None
                    else settings.snowflake_embedding_model
                )
                embedder = SnowflakeAiEmbedder(
                    settings.snowflake_connection_name,
                    embedding_model,
                )
                resources.callback(embedder.close)

            if "semantic" in retrieval_modes:
                if embedder is None:
                    raise ValueError("semantic retrieval requires an embedder")
                embeddings = embedder.embed_texts(
                    [chunk.text for chunk in store.chunks]
                )
                vector_store = ChromaVectorStore(
                    store.manifest.claim_id,
                    staging_root / "index" / "chroma",
                )
                try:
                    vector_store.index_chunks(store.chunks, embeddings)
                finally:
                    vector_store.close()

            if "lightrag" in retrieval_modes:
                if selected_spec is None:
                    raise ValueError("LightRAG retrieval requires an embedding spec")
                runtime = resources.enter_context(open_agent_runtime(llm_settings))
                index_lightrag_chunks(
                    runtime,
                    llm_settings,
                    staging_root / "index" / "lightrag",
                    store.manifest.claim_id,
                    store.chunks,
                    selected_spec,
                    snowflake_embedder=embedder,
                    seed_cache_path=_lightrag_seed_cache_path(
                        claim_path,
                        rebuild_cache_path,
                    ),
                )

        if selected_spec is not None:
            embedding_provider = selected_spec.provider
            embedding_model = selected_spec.model
        elif "semantic" in retrieval_modes:
            embedding_provider = "snowflake"
            embedding_model = settings.snowflake_embedding_model
        else:
            embedding_provider = None
            embedding_model = None

        manifest = ClaimManifest.model_validate(
            store.manifest.model_dump()
            | {
                "retrieval_mode": retrieval_modes[0],
                "additional_retrieval_modes": list(retrieval_modes[1:]),
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
            }
        )
        if "lightrag" in retrieval_modes:
            validate_lightrag_index(
                staging_root / "index" / "lightrag",
                manifest,
            )

        _commit_rebuilt_index(claim_path, staging_root / "index", manifest)
        rebuild_cache_path.unlink(missing_ok=True)
        return manifest
    except Exception:
        _preserve_lightrag_rebuild_cache(staging_root, rebuild_cache_path)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _lightrag_seed_cache_path(
    claim_path: Path,
    rebuild_cache_path: Path,
) -> Path:
    if rebuild_cache_path.exists():
        return rebuild_cache_path
    return claim_path / "index" / "lightrag" / LLM_CACHE_FILE


def _preserve_lightrag_rebuild_cache(
    staging_root: Path,
    rebuild_cache_path: Path,
) -> None:
    staged_cache = staging_root / "index" / "lightrag" / LLM_CACHE_FILE
    if not staged_cache.exists():
        return
    temporary_path = rebuild_cache_path.with_suffix(rebuild_cache_path.suffix + ".tmp")
    shutil.copy2(staged_cache, temporary_path)
    temporary_path.replace(rebuild_cache_path)


def _retrieval_modes(
    knowledge_base: KnowledgeBaseEngine,
    custom_mode: RetrievalMode,
) -> tuple[RetrievalMode, ...]:
    if knowledge_base == "custom":
        return (custom_mode,)
    if knowledge_base == "lightrag":
        return ("lightrag",)
    if knowledge_base == "both":
        return (custom_mode, "lightrag")
    raise ValueError(f"Unknown knowledge-base engine: {knowledge_base}")


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
