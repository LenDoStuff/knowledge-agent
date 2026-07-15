"""Embedded LightRAG indexing and retrieval for persisted claim chunks."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Literal, Sequence

import networkx as nx
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import finalize_share_data
from lightrag.utils import EmbeddingFunc
from openai import APIStatusError
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunUsage,
    SystemPromptPart,
    TextPart,
    UsageLimits,
    UserPromptPart,
)

from knowledge_agent.claims.embeddings import TextEmbedder
from knowledge_agent.claims.filesystem import read_json, write_json
from knowledge_agent.claims.models import ClaimManifest, DocumentChunk, utc_now
from knowledge_agent.llm.config import LlmSettings, llm_provider
from knowledge_agent.llm.providers import AgentRuntime


LIGHTRAG_VERSION = "1.5.4"
LIGHTRAG_QUERY_MODE = "hybrid"
NVIDIA_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
NVIDIA_EMBEDDING_DIMENSION = 1024
NVIDIA_EMBEDDING_MAX_TOKENS = 8192
METADATA_FILE = "metadata.json"
GRAPH_FILE = "graph_chunk_entity_relation.graphml"
DOC_STATUS_FILE = "kv_store_doc_status.json"
LLM_CACHE_FILE = "kv_store_llm_response_cache.json"
NVIDIA_EMBEDDING_BATCH_SIZE = 32
NVIDIA_EMBEDDING_MAX_RETRIES = 3
NVIDIA_EMBEDDING_RETRY_STATUSES = {429, 500, 502, 503, 504}
NVIDIA_LLM_MIN_INTERVAL_SECONDS = 10.0
_LIGHTRAG_RESOURCE_LOCK = RLock()
LOGGER = logging.getLogger(__name__)
NvidiaEmbeddingInputType = Literal["passage", "query"]

SNOWFLAKE_EMBEDDING_SPECS: dict[str, tuple[int, int]] = {
    "snowflake-arctic-embed-l-v2.0": (1024, 512),
    "snowflake-arctic-embed-l-v2.0-8k": (1024, 8192),
    "nv-embed-qa-4": (1024, 512),
    "multilingual-e5-large": (1024, 512),
    "voyage-multilingual-2": (1024, 32000),
    "snowflake-arctic-embed-m-v1.5": (768, 512),
    "snowflake-arctic-embed-m": (768, 512),
    "e5-base-v2": (768, 512),
}


class UsageSnapshot(BaseModel):
    requests: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class LightRagIndexMetadata(BaseModel):
    schema_version: Literal[1] = 1
    claim_id: str = Field(min_length=1)
    lightrag_version: Literal[LIGHTRAG_VERSION] = LIGHTRAG_VERSION
    llm_provider: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    embedding_max_tokens: int = Field(gt=0)
    query_mode: Literal["hybrid"] = LIGHTRAG_QUERY_MODE
    indexed_chunk_count: int = Field(ge=0)
    entity_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    indexing_usage: UsageSnapshot = Field(default_factory=UsageSnapshot)
    created_at: datetime = Field(default_factory=utc_now)


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    dimension: int
    max_tokens: int


@dataclass(frozen=True)
class LightRagGraph:
    entities: list[dict[str, object]]
    relationships: list[dict[str, object]]


class _PydanticLightRagModel:
    """Adapt LightRAG prompt calls to the selected PydanticAI model."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        min_request_interval_seconds: float = 0.0,
    ) -> None:
        self.runtime = runtime
        self.usage = RunUsage()
        self.usage_limits: UsageLimits | None = None
        self.min_request_interval_seconds = min_request_interval_seconds
        self._request_lock = asyncio.Lock()
        self._last_request_started: float | None = None

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        response_format: object | None = None,
        **_kwargs: object,
    ) -> str:
        output_type: type[str] | Any = (
            dict[str, Any] if response_format is not None else str
        )
        agent = Agent(
            model=self.runtime.model,
            output_type=output_type,
            name="lightrag_model",
            retries=0,
        )
        model_settings = {"max_tokens": max_tokens} if max_tokens else None
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            if self._last_request_started is not None:
                delay = self.min_request_interval_seconds - (
                    loop.time() - self._last_request_started
                )
                if delay > 0:
                    LOGGER.info(
                        "lightrag_llm_request_pacing delay_seconds=%.2f",
                        delay,
                    )
                    await asyncio.sleep(delay)
            self._last_request_started = loop.time()
            result = await agent.run(
                prompt,
                instructions=system_prompt,
                message_history=_model_history(history_messages or ()),
                usage=self.usage,
                usage_limits=self.usage_limits,
                model_settings=model_settings,
            )
        if isinstance(result.output, str):
            return result.output
        return json.dumps(result.output, ensure_ascii=False)


@dataclass
class LightRagResource:
    rag: LightRAG
    model_adapter: _PydanticLightRagModel
    metadata: LightRagIndexMetadata | None = None
    embedding_input_type: ContextVar[NvidiaEmbeddingInputType] | None = None

    def bind_usage(self, usage: RunUsage, limits: UsageLimits) -> None:
        self.model_adapter.usage = usage
        self.model_adapter.usage_limits = limits

    def clear_usage(self) -> None:
        self.model_adapter.usage = RunUsage()
        self.model_adapter.usage_limits = None

    async def retrieve_chunk_ids(self, query: str, top_k: int) -> list[str]:
        token = (
            self.embedding_input_type.set("query")
            if self.embedding_input_type is not None
            else None
        )
        try:
            result = await self.rag.aquery_data(
                query,
                QueryParam(
                    mode=LIGHTRAG_QUERY_MODE,
                    top_k=top_k,
                    chunk_top_k=top_k,
                    enable_rerank=False,
                ),
            )
        finally:
            if self.embedding_input_type is not None and token is not None:
                self.embedding_input_type.reset(token)
        if result.get("status") != "success":
            message = str(result.get("message") or "LightRAG query failed")
            if "no result" in message.casefold() or "no relevant" in message.casefold():
                return []
            raise RuntimeError(message)
        data = result.get("data")
        if not isinstance(data, dict):
            raise ValueError("LightRAG query result has no data object")
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("LightRAG query result has no chunks list")

        chunk_ids: list[str] = []
        seen: set[str] = set()
        for item in chunks:
            if not isinstance(item, dict):
                raise ValueError("LightRAG returned a malformed chunk")
            chunk_id = item.get("file_path")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("LightRAG returned a chunk without a file path")
            if chunk_id != Path(chunk_id).name:
                raise ValueError(
                    f"LightRAG returned a non-local chunk identifier: {chunk_id}"
                )
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
            if len(chunk_ids) == top_k:
                break
        return chunk_ids

    async def close(self) -> None:
        try:
            await self.rag.finalize_storages()
        finally:
            finalize_share_data()


def embedding_spec(
    settings: LlmSettings,
    snowflake_model: str,
) -> EmbeddingSpec:
    if settings.profile == "api_key":
        return EmbeddingSpec(
            provider="nvidia",
            model=NVIDIA_EMBEDDING_MODEL,
            dimension=NVIDIA_EMBEDDING_DIMENSION,
            max_tokens=NVIDIA_EMBEDDING_MAX_TOKENS,
        )
    spec = SNOWFLAKE_EMBEDDING_SPECS.get(snowflake_model)
    if spec is None:
        raise ValueError(
            "LightRAG does not know the dimension and context size for "
            f"Snowflake embedding model {snowflake_model!r}"
        )
    return EmbeddingSpec(
        provider="snowflake",
        model=snowflake_model,
        dimension=spec[0],
        max_tokens=spec[1],
    )


def index_lightrag_chunks(
    runtime: AgentRuntime,
    settings: LlmSettings,
    index_path: Path,
    claim_id: str,
    chunks: Sequence[DocumentChunk],
    spec: EmbeddingSpec,
    *,
    snowflake_embedder: TextEmbedder | None = None,
    seed_cache_path: Path | None = None,
) -> LightRagIndexMetadata:
    with _LIGHTRAG_RESOURCE_LOCK:
        return runtime.run_coroutine(
            _index_lightrag_chunks(
                runtime,
                settings,
                index_path,
                claim_id,
                chunks,
                spec,
                snowflake_embedder=snowflake_embedder,
                seed_cache_path=seed_cache_path,
            )
        )


async def _index_lightrag_chunks(
    runtime: AgentRuntime,
    settings: LlmSettings,
    index_path: Path,
    claim_id: str,
    chunks: Sequence[DocumentChunk],
    spec: EmbeddingSpec,
    *,
    snowflake_embedder: TextEmbedder | None,
    seed_cache_path: Path | None,
) -> LightRagIndexMetadata:
    if index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True)
    if seed_cache_path is not None and seed_cache_path.exists():
        shutil.copy2(seed_cache_path, index_path / LLM_CACHE_FILE)
    resource = await create_lightrag_resource(
        runtime,
        settings,
        index_path,
        spec,
        snowflake_embedder=snowflake_embedder,
    )
    try:
        if chunks:
            document_ids = [chunk.source_ref for chunk in chunks]
            await resource.rag.ainsert(
                [chunk.text for chunk in chunks],
                ids=document_ids,
                file_paths=[chunk.chunk_id for chunk in chunks],
            )
            statuses = await resource.rag.aget_docs_by_ids(document_ids)
            incomplete = [
                (document_id, statuses.get(document_id))
                for document_id in document_ids
                if _document_status(statuses.get(document_id))
                != DocStatus.PROCESSED.value
            ]
            if incomplete:
                document_list = ", ".join(item[0] for item in incomplete)
                error_details = [
                    f"{document_id}: {error}"
                    for document_id, record in incomplete
                    if (error := _document_error(record)) is not None
                ]
                message = "LightRAG indexing did not complete for: " + document_list
                if error_details:
                    message += ". Errors: " + "; ".join(error_details)
                raise RuntimeError(message)
        nodes = await resource.rag.chunk_entity_relation_graph.get_all_nodes()
        edges = await resource.rag.chunk_entity_relation_graph.get_all_edges()
        metadata = LightRagIndexMetadata(
            claim_id=claim_id,
            llm_provider=llm_provider(settings),
            llm_model=settings.model,
            embedding_provider=spec.provider,
            embedding_model=spec.model,
            embedding_dimension=spec.dimension,
            embedding_max_tokens=spec.max_tokens,
            indexed_chunk_count=len(chunks),
            entity_count=len(nodes),
            relationship_count=len(edges),
            indexing_usage=_usage_snapshot(resource.model_adapter.usage),
        )
        write_json(index_path / METADATA_FILE, metadata.model_dump(mode="json"))
        return metadata
    finally:
        await resource.close()


async def create_lightrag_resource(
    runtime: AgentRuntime,
    settings: LlmSettings,
    index_path: Path,
    spec: EmbeddingSpec,
    *,
    snowflake_embedder: TextEmbedder | None = None,
    metadata: LightRagIndexMetadata | None = None,
) -> LightRagResource:
    if version("lightrag-hku") != LIGHTRAG_VERSION:
        raise RuntimeError(f"LightRAG {LIGHTRAG_VERSION} is required")
    if spec.provider == "snowflake" and snowflake_embedder is None:
        raise ValueError("Snowflake LightRAG requires a Snowflake embedder")

    model_adapter = _PydanticLightRagModel(
        runtime,
        min_request_interval_seconds=(
            NVIDIA_LLM_MIN_INTERVAL_SECONDS if settings.profile == "api_key" else 0.0
        ),
    )
    embedding_input_type = ContextVar[NvidiaEmbeddingInputType](
        "lightrag_embedding_input_type",
        default="passage",
    )

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        response_format: object | None = None,
        **kwargs: object,
    ) -> str:
        return await model_adapter.complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    async def embed(texts: list[str]) -> np.ndarray:
        if spec.provider == "nvidia":
            response = await _create_nvidia_embeddings(
                runtime,
                spec,
                texts,
                embedding_input_type.get(),
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return np.asarray([item.embedding for item in ordered], dtype=float)
        assert snowflake_embedder is not None
        values = await asyncio.to_thread(snowflake_embedder.embed_texts, texts)
        return np.asarray(values, dtype=float)

    rag: LightRAG | None = None
    try:
        rag = LightRAG(
            working_dir=str(index_path),
            embedding_func=EmbeddingFunc(
                embedding_dim=spec.dimension,
                max_token_size=spec.max_tokens,
                model_name=spec.model,
                func=embed,
            ),
            embedding_batch_num=(
                NVIDIA_EMBEDDING_BATCH_SIZE if spec.provider == "nvidia" else 10
            ),
            embedding_func_max_async=1,
            llm_model_func=complete,
            llm_model_name=settings.model,
            llm_model_max_async=1,
            max_parallel_insert=1,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
        )
        await rag.initialize_storages()
    except Exception:
        try:
            if rag is not None:
                await rag.finalize_storages()
        finally:
            finalize_share_data()
        raise
    return LightRagResource(
        rag=rag,
        model_adapter=model_adapter,
        metadata=metadata,
        embedding_input_type=embedding_input_type,
    )


async def _create_nvidia_embeddings(
    runtime: AgentRuntime,
    spec: EmbeddingSpec,
    texts: list[str],
    input_type: NvidiaEmbeddingInputType,
) -> CreateEmbeddingResponse:
    for retry_number in range(NVIDIA_EMBEDDING_MAX_RETRIES + 1):
        try:
            return await runtime.openai.embeddings.create(
                model=spec.model,
                input=texts,
                dimensions=spec.dimension,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
        except APIStatusError as exc:
            if (
                exc.status_code not in NVIDIA_EMBEDDING_RETRY_STATUSES
                or retry_number == NVIDIA_EMBEDDING_MAX_RETRIES
            ):
                raise
            delay_seconds = 2**retry_number
            LOGGER.warning(
                "nvidia_embedding_retry retry=%s max_retries=%s status=%s "
                "delay_seconds=%s batch_size=%s",
                retry_number + 1,
                NVIDIA_EMBEDDING_MAX_RETRIES,
                exc.status_code,
                delay_seconds,
                len(texts),
            )
            await asyncio.sleep(delay_seconds)
    raise AssertionError("unreachable")


@contextmanager
def open_lightrag_resource(
    runtime: AgentRuntime,
    settings: LlmSettings,
    index_path: Path,
    spec: EmbeddingSpec,
    *,
    snowflake_embedder: TextEmbedder | None = None,
    metadata: LightRagIndexMetadata | None = None,
) -> Iterator[LightRagResource]:
    with _LIGHTRAG_RESOURCE_LOCK:
        resource = runtime.run_coroutine(
            create_lightrag_resource(
                runtime,
                settings,
                index_path,
                spec,
                snowflake_embedder=snowflake_embedder,
                metadata=metadata,
            )
        )
        try:
            yield resource
        finally:
            runtime.run_coroutine(resource.close())


def load_lightrag_metadata(index_path: Path) -> LightRagIndexMetadata:
    data = read_json(index_path / METADATA_FILE)
    if not isinstance(data, dict):
        raise ValueError("LightRAG metadata must contain a JSON object")
    return LightRagIndexMetadata.model_validate(data)


def validate_lightrag_index(
    index_path: Path,
    manifest: ClaimManifest,
) -> LightRagIndexMetadata:
    metadata = load_lightrag_metadata(index_path)
    if metadata.claim_id != manifest.claim_id:
        raise ValueError("LightRAG metadata claim_id does not match manifest")
    if metadata.indexed_chunk_count != manifest.chunk_count:
        raise ValueError("LightRAG indexed chunk count does not match manifest")
    if metadata.embedding_provider != manifest.embedding_provider:
        raise ValueError("LightRAG embedding provider does not match manifest")
    if metadata.embedding_model != manifest.embedding_model:
        raise ValueError("LightRAG embedding model does not match manifest")
    if not (index_path / GRAPH_FILE).exists():
        raise FileNotFoundError("LightRAG graph index is missing")
    _validate_document_statuses(index_path, metadata.indexed_chunk_count)
    return metadata


def _validate_document_statuses(index_path: Path, expected_count: int) -> None:
    if expected_count == 0:
        return
    data = read_json(index_path / DOC_STATUS_FILE)
    if not isinstance(data, dict):
        raise ValueError("LightRAG document status must contain a JSON object")
    if len(data) != expected_count:
        raise ValueError("LightRAG document status count does not match metadata")
    incomplete = [
        document_id
        for document_id, record in data.items()
        if not isinstance(record, dict)
        or record.get("status") != DocStatus.PROCESSED.value
    ]
    if incomplete:
        raise ValueError(
            "LightRAG index contains incomplete documents: " + ", ".join(incomplete)
        )


def _document_status(record: object) -> str | None:
    status = (
        record.get("status")
        if isinstance(record, dict)
        else getattr(record, "status", None)
    )
    if isinstance(status, DocStatus):
        return status.value
    return status if isinstance(status, str) else None


def _document_error(record: object) -> str | None:
    error = (
        record.get("error_msg")
        if isinstance(record, dict)
        else getattr(record, "error_msg", None)
    )
    return error.strip() if isinstance(error, str) and error.strip() else None


def load_lightrag_graph(index_path: Path) -> LightRagGraph:
    graph_path = index_path / GRAPH_FILE
    if not graph_path.exists():
        raise FileNotFoundError("LightRAG graph index is missing")
    graph = nx.read_graphml(graph_path)
    entities = [
        {"Entity": str(node_id), **_display_properties(properties)}
        for node_id, properties in graph.nodes(data=True)
    ]
    relationships = [
        {
            "Source": str(source),
            "Target": str(target),
            **_display_properties(properties),
        }
        for source, target, properties in graph.edges(data=True)
    ]
    return LightRagGraph(entities=entities, relationships=relationships)


def _display_properties(properties: dict[str, Any]) -> dict[str, object]:
    return {
        str(key).replace("_", " ").title(): value
        for key, value in properties.items()
        if key not in {"source_id"}
    }


def _model_history(
    messages: Sequence[dict[str, str]],
) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("LightRAG history message content must be text")
        if role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content)]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content)]))
        elif role == "system":
            history.append(ModelRequest(parts=[SystemPromptPart(content)]))
        else:
            raise ValueError(f"Unsupported LightRAG history role: {role!r}")
    return history


def _usage_snapshot(usage: RunUsage) -> UsageSnapshot:
    return UsageSnapshot(
        requests=usage.requests,
        tool_calls=usage.tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
