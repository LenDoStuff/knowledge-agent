"""Tests for embedded LightRAG indexing, retrieval, and rebuilds."""

import asyncio
import json
import shutil
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import pytest
from lightrag import LightRAG
from lightrag.kg.shared_storage import is_share_data_initialized
from pydantic_ai import RunUsage
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import knowledge_agent.claims.lightrag as lightrag_module
from knowledge_agent.agents.claim_researcher.tools import search_claim_evidence
from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import (
    LIGHTRAG_REBUILD_CACHE_FILE,
    _commit_rebuilt_index,
    rebuild_claim_knowledge_base,
)
from knowledge_agent.claims.lightrag import (
    DOC_STATUS_FILE,
    GRAPH_FILE,
    LLM_CACHE_FILE,
    METADATA_FILE,
    EmbeddingSpec,
    LightRagIndexMetadata,
    LightRagResource,
    _PydanticLightRagModel,
    create_lightrag_resource,
    embedding_spec,
    index_lightrag_chunks,
    load_lightrag_graph,
    validate_lightrag_index,
)
from knowledge_agent.claims.models import ClaimManifest
from knowledge_agent.claims.store import load_claim_store
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import AgentRuntime


SAMPLE_OUTPUT = Path("examples/claims/sample_output")


class FakeGraphStorage:
    async def get_all_nodes(self):
        return [{"id": "Acme"}, {"id": "Repair Co"}]

    async def get_all_edges(self):
        return [{"source": "Acme", "target": "Repair Co"}]


class FakeIndexRag:
    def __init__(
        self,
        status: str = "processed",
        error_message: str | None = None,
    ) -> None:
        self.chunk_entity_relation_graph = FakeGraphStorage()
        self.inserted: dict[str, object] | None = None
        self.status = status
        self.error_message = error_message

    async def ainsert(self, texts, ids, file_paths):
        self.inserted = {"texts": texts, "ids": ids, "file_paths": file_paths}

    async def aget_docs_by_ids(self, ids):
        return {
            document_id: {
                "status": self.status,
                "error_msg": self.error_message,
            }
            for document_id in ids
        }


class FakeIndexResource:
    def __init__(
        self,
        status: str = "processed",
        error_message: str | None = None,
    ) -> None:
        self.rag = FakeIndexRag(status, error_message)
        self.model_adapter = SimpleNamespace(usage=RunUsage(requests=3))
        self.closed = False

    async def close(self):
        self.closed = True


class FakeQueryRag:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = []

    async def aquery_data(self, query, param):
        self.calls.append((query, param))
        return self.payload


def api_settings() -> LlmSettings:
    return LlmSettings(
        profile="api_key",
        model="provider/model",
        reasoning_effort="low",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_api_key_ds4="secret",
    )


def claim_settings(tmp_path: Path) -> ClaimSettings:
    return ClaimSettings(
        data_root=tmp_path,
        document_intelligence_endpoint=None,
        snowflake_connection_name="default",
        snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
    )


def test_embedding_specs_reuse_current_profiles():
    nvidia = embedding_spec(api_settings(), "ignored")
    assert (nvidia.provider, nvidia.model, nvidia.dimension) == (
        "nvidia",
        "nvidia/llama-nemotron-embed-1b-v2",
        1024,
    )

    azure = LlmSettings(
        profile="azure_project",
        model="deployment",
        reasoning_effort="low",
        azure_ai_project_endpoint="https://project.example",
    )
    snowflake = embedding_spec(azure, "snowflake-arctic-embed-l-v2.0")
    assert (snowflake.provider, snowflake.dimension, snowflake.max_tokens) == (
        "snowflake",
        1024,
        512,
    )
    with pytest.raises(ValueError, match="does not know"):
        embedding_spec(azure, "unknown-model")


def test_lightrag_llm_calls_are_routed_through_pydanticai():
    observed_messages = []

    def model_function(messages, info: AgentInfo):
        observed_messages.extend(messages)
        return ModelResponse(parts=[TextPart("extracted entities")])

    runner = asyncio.Runner()
    try:
        runtime = AgentRuntime(
            model=FunctionModel(model_function),
            runner=runner,
            openai=cast(Any, None),
        )
        adapter = _PydanticLightRagModel(runtime)
        result = runner.run(
            adapter.complete(
                "Extract this claim.",
                system_prompt="Return entity records.",
                history_messages=[
                    {"role": "user", "content": "Earlier input"},
                    {"role": "assistant", "content": "Earlier output"},
                ],
            )
        )
    finally:
        runner.close()

    assert result == "extracted entities"
    assert adapter.usage.requests == 1
    assert any(
        part.content == "Earlier input"
        for message in observed_messages
        for part in message.parts
        if hasattr(part, "content")
    )


def test_lightrag_json_requests_use_pydanticai_structured_output():
    def model_function(messages, info: AgentInfo):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "response": {
                            "high_level_keywords": ["repairs"],
                            "low_level_keywords": ["bumper"],
                        }
                    },
                )
            ]
        )

    runner = asyncio.Runner()
    try:
        adapter = _PydanticLightRagModel(
            AgentRuntime(
                model=FunctionModel(model_function),
                runner=runner,
                openai=cast(Any, None),
            )
        )
        payload = json.loads(
            runner.run(
                adapter.complete(
                    "Extract keywords.",
                    response_format={"type": "json_object"},
                )
            )
        )
    finally:
        runner.close()

    assert payload == {
        "high_level_keywords": ["repairs"],
        "low_level_keywords": ["bumper"],
    }


def test_lightrag_llm_calls_are_explicitly_paced(monkeypatch):
    sleeps = []

    def model_function(messages, info: AgentInfo):
        return ModelResponse(parts=[TextPart("entity records")])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(lightrag_module.asyncio, "sleep", fake_sleep)
    runner = asyncio.Runner()
    try:
        adapter = _PydanticLightRagModel(
            AgentRuntime(
                model=FunctionModel(model_function),
                runner=runner,
                openai=cast(Any, None),
            ),
            min_request_interval_seconds=10.0,
        )

        async def exercise():
            await adapter.complete("First document")
            await adapter.complete("Second document")

        runner.run(exercise())
    finally:
        runner.close()

    assert len(sleeps) == 1
    assert 9.0 < sleeps[0] <= 10.0


def test_resource_callback_and_shared_state_are_safe_across_event_loops(
    monkeypatch,
    tmp_path,
):
    async def no_op_storage(_self):
        pass

    monkeypatch.setattr(LightRAG, "initialize_storages", no_op_storage)
    monkeypatch.setattr(LightRAG, "finalize_storages", no_op_storage)
    runtime = SimpleNamespace(
        model=object(),
        openai=SimpleNamespace(lock=threading.RLock()),
    )

    async def exercise(index_path):
        resource = await create_lightrag_resource(
            cast(Any, runtime),
            api_settings(),
            index_path,
            EmbeddingSpec("nvidia", "baai/bge-m3", 1024, 8192),
        )
        await resource.close()
        return resource

    first = asyncio.run(exercise(tmp_path / "first"))
    assert not is_share_data_initialized()
    second = asyncio.run(exercise(tmp_path / "second"))

    assert not is_share_data_initialized()
    assert not hasattr(first.rag.llm_model_func, "__self__")
    assert not hasattr(second.rag.llm_model_func, "__self__")


def test_nvidia_embeddings_reuse_the_runtime_openai_client(monkeypatch, tmp_path):
    captured = {"requests": []}

    class FakeEmbeddings:
        async def create(self, **kwargs):
            captured["requests"].append(kwargs)
            first = [0.0] * 1024
            second = [0.0] * 1024
            first[0] = 1.0
            second[1] = 1.0
            if len(kwargs["input"]) == 1:
                return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=first)])
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=second),
                    SimpleNamespace(index=0, embedding=first),
                ]
            )

    class FakeLightRag:
        def __init__(self, **kwargs):
            captured["rag"] = kwargs
            self.embedding_func = kwargs["embedding_func"]

        async def initialize_storages(self):
            pass

        async def finalize_storages(self):
            pass

        async def aquery_data(self, query, _params):
            await self.embedding_func([query])
            return {
                "status": "success",
                "data": {"chunks": [{"file_path": "CHUNK-1"}]},
            }

    monkeypatch.setattr("knowledge_agent.claims.lightrag.LightRAG", FakeLightRag)
    runtime = SimpleNamespace(
        model=object(),
        openai=SimpleNamespace(embeddings=FakeEmbeddings()),
    )

    async def exercise():
        resource = await create_lightrag_resource(
            cast(Any, runtime),
            api_settings(),
            tmp_path,
            EmbeddingSpec(
                "nvidia",
                "nvidia/llama-nemotron-embed-1b-v2",
                1024,
                8192,
            ),
        )
        values = await captured["rag"]["embedding_func"](["first", "second"])
        chunk_ids = await resource.retrieve_chunk_ids("fire origin", 1)
        await resource.close()
        return values, chunk_ids

    values, chunk_ids = asyncio.run(exercise())

    assert values.shape == (2, 1024)
    assert values[0, 0] == 1.0
    assert values[1, 1] == 1.0
    assert chunk_ids == ["CHUNK-1"]
    assert captured["requests"][0] == {
        "model": "nvidia/llama-nemotron-embed-1b-v2",
        "input": ["first", "second"],
        "dimensions": 1024,
        "encoding_format": "float",
        "extra_body": {"input_type": "passage", "truncate": "END"},
    }
    assert captured["requests"][1] == {
        "model": "nvidia/llama-nemotron-embed-1b-v2",
        "input": ["fire origin"],
        "dimensions": 1024,
        "encoding_format": "float",
        "extra_body": {"input_type": "query", "truncate": "END"},
    }
    assert captured["rag"]["embedding_batch_num"] == 32
    assert captured["rag"]["embedding_func_max_async"] == 1
    assert captured["rag"]["llm_model_max_async"] == 1
    assert captured["rag"]["max_parallel_insert"] == 1


def test_nvidia_embeddings_retry_transient_statuses_with_bounded_backoff(
    monkeypatch,
    caplog,
):
    calls = []
    sleeps = []

    class FakeStatusError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    class FakeEmbeddings:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= 2:
                raise FakeStatusError(500 if len(calls) == 1 else 429)
            return SimpleNamespace(data=[])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(lightrag_module, "APIStatusError", FakeStatusError)
    monkeypatch.setattr(lightrag_module.asyncio, "sleep", fake_sleep)
    runtime = SimpleNamespace(
        openai=SimpleNamespace(embeddings=FakeEmbeddings()),
    )

    result = asyncio.run(
        lightrag_module._create_nvidia_embeddings(
            cast(Any, runtime),
            EmbeddingSpec("nvidia", "baai/bge-m3", 1024, 8192),
            ["evidence"],
            "passage",
        )
    )

    assert result.data == []
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert "nvidia_embedding_retry" in caplog.text


@pytest.mark.parametrize("status_code, expected_calls", [(400, 1), (500, 4)])
def test_nvidia_embedding_retries_are_status_filtered_and_capped(
    monkeypatch,
    status_code,
    expected_calls,
):
    calls = 0
    sleeps = []

    class FakeStatusError(Exception):
        def __init__(self):
            self.status_code = status_code

    class FakeEmbeddings:
        async def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise FakeStatusError()

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(lightrag_module, "APIStatusError", FakeStatusError)
    monkeypatch.setattr(lightrag_module.asyncio, "sleep", fake_sleep)
    runtime = SimpleNamespace(
        openai=SimpleNamespace(embeddings=FakeEmbeddings()),
    )

    with pytest.raises(FakeStatusError):
        asyncio.run(
            lightrag_module._create_nvidia_embeddings(
                cast(Any, runtime),
                EmbeddingSpec("nvidia", "baai/bge-m3", 1024, 8192),
                ["evidence"],
                "passage",
            )
        )

    assert calls == expected_calls
    assert sleeps == ([1, 2, 4] if status_code == 500 else [])


def test_indexing_preserves_chunk_ids_and_source_refs(monkeypatch, tmp_path):
    store = load_claim_store(SAMPLE_OUTPUT)
    resource = FakeIndexResource()

    async def fake_create(*args, **kwargs):
        return resource

    monkeypatch.setattr(
        "knowledge_agent.claims.lightrag.create_lightrag_resource",
        fake_create,
    )
    runtime = SimpleNamespace(run_coroutine=lambda coroutine: asyncio.run(coroutine))
    metadata = index_lightrag_chunks(
        cast(Any, runtime),
        api_settings(),
        tmp_path / "lightrag",
        store.manifest.claim_id,
        store.chunks,
        EmbeddingSpec("nvidia", "baai/bge-m3", 1024, 8192),
    )

    assert resource.closed
    assert resource.rag.inserted == {
        "texts": [chunk.text for chunk in store.chunks],
        "ids": [chunk.source_ref for chunk in store.chunks],
        "file_paths": [chunk.chunk_id for chunk in store.chunks],
    }
    assert metadata.indexed_chunk_count == len(store.chunks)
    assert metadata.entity_count == 2
    assert metadata.relationship_count == 1
    assert metadata.indexing_usage.requests == 3
    saved = LightRagIndexMetadata.model_validate_json(
        (tmp_path / "lightrag" / METADATA_FILE).read_text(encoding="utf-8")
    )
    assert saved == metadata


def test_indexing_rejects_incomplete_lightrag_documents(monkeypatch, tmp_path):
    store = load_claim_store(SAMPLE_OUTPUT)
    resource = FakeIndexResource(
        status="failed",
        error_message="provider returned HTTP 429",
    )

    async def fake_create(*args, **kwargs):
        return resource

    monkeypatch.setattr(
        "knowledge_agent.claims.lightrag.create_lightrag_resource",
        fake_create,
    )
    runtime = SimpleNamespace(run_coroutine=lambda coroutine: asyncio.run(coroutine))
    index_path = tmp_path / "lightrag"

    with pytest.raises(
        RuntimeError,
        match="indexing did not complete.*provider returned HTTP 429",
    ):
        index_lightrag_chunks(
            cast(Any, runtime),
            api_settings(),
            index_path,
            store.manifest.claim_id,
            store.chunks,
            EmbeddingSpec("nvidia", "baai/bge-m3", 1024, 8192),
        )

    assert resource.closed
    assert not (index_path / METADATA_FILE).exists()


def test_indexing_can_seed_the_existing_claim_llm_cache(monkeypatch, tmp_path):
    store = load_claim_store(SAMPLE_OUTPUT)
    resource = FakeIndexResource()
    seed_cache = tmp_path / "old" / LLM_CACHE_FILE
    seed_cache.parent.mkdir()
    seed_cache.write_text('{"cached":"response"}', encoding="utf-8")
    observed_cache = None

    async def fake_create(*args, **kwargs):
        nonlocal observed_cache
        index_path = args[2]
        observed_cache = (index_path / LLM_CACHE_FILE).read_text(encoding="utf-8")
        return resource

    monkeypatch.setattr(
        "knowledge_agent.claims.lightrag.create_lightrag_resource",
        fake_create,
    )
    runtime = SimpleNamespace(run_coroutine=lambda coroutine: asyncio.run(coroutine))

    index_lightrag_chunks(
        cast(Any, runtime),
        api_settings(),
        tmp_path / "new",
        store.manifest.claim_id,
        store.chunks,
        EmbeddingSpec("nvidia", "baai/bge-m3", 1024, 8192),
        seed_cache_path=seed_cache,
    )

    assert observed_cache == '{"cached":"response"}'


def test_structured_retrieval_is_ordered_deduplicated_and_hard_limited():
    payload = {
        "status": "success",
        "message": "ok",
        "data": {
            "chunks": [
                {"file_path": "CHUNK-2"},
                {"file_path": "CHUNK-1"},
                {"file_path": "CHUNK-2"},
                {"file_path": "CHUNK-3"},
            ]
        },
    }
    rag = FakeQueryRag(payload)
    adapter = SimpleNamespace(usage=RunUsage(), usage_limits=None)
    resource = LightRagResource(cast(Any, rag), cast(Any, adapter))

    chunk_ids = asyncio.run(resource.retrieve_chunk_ids("repair", 2))

    assert chunk_ids == ["CHUNK-2", "CHUNK-1"]
    query, param = rag.calls[0]
    assert query == "repair"
    assert param.mode == "hybrid"
    assert param.top_k == param.chunk_top_k == 2
    assert param.enable_rerank is False


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"status": "failure", "message": "provider unavailable", "data": {}},
            "provider unavailable",
        ),
        (
            {"status": "success", "message": "ok", "data": {"chunks": [{}]}},
            "without a file path",
        ),
        (
            {
                "status": "success",
                "message": "ok",
                "data": {"chunks": [{"file_path": "folder/CHUNK-1"}]},
            },
            "non-local",
        ),
    ],
)
def test_structured_retrieval_surfaces_failures(payload, message):
    resource = LightRagResource(
        cast(Any, FakeQueryRag(payload)),
        cast(Any, SimpleNamespace(usage=RunUsage(), usage_limits=None)),
    )
    with pytest.raises((RuntimeError, ValueError), match=message):
        asyncio.run(resource.retrieve_chunk_ids("repair", 8))


def test_lightrag_evidence_maps_back_to_existing_claim_citations():
    store = load_claim_store(SAMPLE_OUTPUT)
    selected = [store.chunks[1].chunk_id, store.chunks[0].chunk_id]

    class Retriever:
        async def retrieve_chunk_ids(self, query, top_k):
            return selected

    manifest = ClaimManifest.model_validate(
        store.manifest.model_dump()
        | {
            "retrieval_mode": "lightrag",
            "embedding_provider": "nvidia",
            "embedding_model": "baai/bge-m3",
        }
    )
    lightrag_store = replace(
        store,
        manifest=manifest,
        retrieval_mode="lightrag",
        lightrag=cast(Any, Retriever()),
    )

    evidence = asyncio.run(search_claim_evidence(lightrag_store, "repair", 8))

    assert [item.source_ref for item in evidence] == [
        store.chunks_by_id[chunk_id].source_ref for chunk_id in selected
    ]
    assert [item.text for item in evidence] == [
        store.chunks_by_id[chunk_id].text for chunk_id in selected
    ]


def test_index_validation_and_graph_tables(tmp_path):
    source = load_claim_store(SAMPLE_OUTPUT)
    manifest = ClaimManifest.model_validate(
        source.manifest.model_dump()
        | {
            "retrieval_mode": "lightrag",
            "embedding_provider": "nvidia",
            "embedding_model": "baai/bge-m3",
        }
    )
    index_path = tmp_path / "index" / "lightrag"
    _write_fake_index(index_path, manifest)

    metadata = validate_lightrag_index(index_path, manifest)
    graph = load_lightrag_graph(index_path)

    assert metadata.indexed_chunk_count == manifest.chunk_count
    assert graph.entities[0]["Entity"] == "Acme"
    assert graph.relationships[0]["Source"] == "Acme"
    assert graph.relationships[0]["Target"] == "Repair Co"

    broken = manifest.model_copy(update={"embedding_model": "different"})
    with pytest.raises(ValueError, match="embedding model"):
        validate_lightrag_index(index_path, broken)

    statuses = json.loads((index_path / DOC_STATUS_FILE).read_text(encoding="utf-8"))
    statuses[next(iter(statuses))]["status"] = "failed"
    (index_path / DOC_STATUS_FILE).write_text(json.dumps(statuses), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete documents"):
        validate_lightrag_index(index_path, manifest)


def test_rebuild_switches_engines_without_changing_claim_artifacts(
    monkeypatch, tmp_path
):
    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    history_path = claim_path / "research" / "history.json"
    history_path.parent.mkdir()
    history_path.write_text('{"audit":"preserved"}', encoding="utf-8")
    original_chunks = (claim_path / "chunks.jsonl").read_bytes()

    @contextmanager
    def fake_runtime(_settings):
        yield object()

    def fake_index(runtime, settings, index_path, claim_id, chunks, spec, **kwargs):
        manifest_data = json.loads((claim_path / "manifest.json").read_text())
        manifest = ClaimManifest.model_validate(
            manifest_data
            | {
                "retrieval_mode": "lightrag",
                "additional_retrieval_modes": [],
                "embedding_provider": spec.provider,
                "embedding_model": spec.model,
            }
        )
        _write_fake_index(index_path, manifest, llm_model=settings.model)

    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.open_agent_runtime", fake_runtime
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.index_lightrag_chunks", fake_index
    )

    both_manifest = rebuild_claim_knowledge_base(
        claim_path,
        "both",
        claim_settings(tmp_path),
        api_settings(),
    )
    assert both_manifest.available_retrieval_modes == ("lexical", "lightrag")
    assert (claim_path / "index" / "lightrag" / METADATA_FILE).exists()

    lightrag_manifest = rebuild_claim_knowledge_base(
        claim_path,
        "lightrag",
        claim_settings(tmp_path),
        api_settings(),
    )
    assert lightrag_manifest.retrieval_mode == "lightrag"
    assert lightrag_manifest.additional_retrieval_modes == []
    assert (claim_path / "index" / "lightrag" / METADATA_FILE).exists()
    assert (claim_path / "chunks.jsonl").read_bytes() == original_chunks
    assert history_path.read_text(encoding="utf-8") == '{"audit":"preserved"}'

    status_path = claim_path / "index" / "lightrag" / DOC_STATUS_FILE
    statuses = json.loads(status_path.read_text(encoding="utf-8"))
    statuses[next(iter(statuses))]["status"] = "failed"
    status_path.write_text(json.dumps(statuses), encoding="utf-8")
    custom_manifest = rebuild_claim_knowledge_base(
        claim_path,
        "custom",
        claim_settings(tmp_path),
        api_settings(),
    )
    assert custom_manifest.retrieval_mode == "lexical"
    assert not (claim_path / "index").exists()
    assert (claim_path / "chunks.jsonl").read_bytes() == original_chunks
    assert history_path.read_text(encoding="utf-8") == '{"audit":"preserved"}'


def test_failed_rebuild_preserves_only_the_staged_llm_cache(monkeypatch, tmp_path):
    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    active_cache = claim_path / "index" / "lightrag" / LLM_CACHE_FILE
    active_cache.parent.mkdir(parents=True)
    active_cache.write_text('{"old":"response"}', encoding="utf-8")
    rebuild_cache = claim_path / LIGHTRAG_REBUILD_CACHE_FILE
    calls = 0

    @contextmanager
    def fake_runtime(_settings):
        yield object()

    def fake_index(runtime, settings, index_path, claim_id, chunks, spec, **kwargs):
        nonlocal calls
        calls += 1
        seed_cache_path = kwargs["seed_cache_path"]
        expected_seed = active_cache if calls == 1 else rebuild_cache
        assert seed_cache_path == expected_seed
        assert (
            json.loads(seed_cache_path.read_text(encoding="utf-8"))["old"] == "response"
        )
        index_path.mkdir(parents=True, exist_ok=True)
        staged_cache = index_path / LLM_CACHE_FILE
        if calls == 1:
            staged_cache.write_text(
                '{"old":"response","completed":"cached"}',
                encoding="utf-8",
            )
            raise RuntimeError("provider rate limit")

        shutil.copy2(seed_cache_path, staged_cache)
        manifest = ClaimManifest.model_validate_json(
            (claim_path / "manifest.json").read_text(encoding="utf-8")
        ).model_copy(
            update={
                "retrieval_mode": "lightrag",
                "additional_retrieval_modes": [],
                "embedding_provider": spec.provider,
                "embedding_model": spec.model,
            }
        )
        _write_fake_index(index_path, manifest, llm_model=settings.model)

    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.open_agent_runtime", fake_runtime
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.index_lightrag_chunks", fake_index
    )

    with pytest.raises(RuntimeError, match="provider rate limit"):
        rebuild_claim_knowledge_base(
            claim_path,
            "lightrag",
            claim_settings(tmp_path),
            api_settings(),
        )

    assert active_cache.read_text(encoding="utf-8") == '{"old":"response"}'
    assert (
        json.loads(rebuild_cache.read_text(encoding="utf-8"))["completed"] == "cached"
    )

    manifest = rebuild_claim_knowledge_base(
        claim_path,
        "lightrag",
        claim_settings(tmp_path),
        api_settings(),
    )

    assert manifest.retrieval_mode == "lightrag"
    assert not rebuild_cache.exists()
    assert json.loads(active_cache.read_text(encoding="utf-8"))["completed"] == "cached"


def test_rebuild_commit_restores_previous_index_when_manifest_write_fails(
    monkeypatch, tmp_path
):
    claim_path = tmp_path / "claim"
    current_index = claim_path / "index"
    staged_index = tmp_path / "staged-index"
    current_index.mkdir(parents=True)
    staged_index.mkdir()
    (current_index / "old.txt").write_text("old", encoding="utf-8")
    (staged_index / "new.txt").write_text("new", encoding="utf-8")

    def fail_manifest_write(*_args, **_kwargs):
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.write_claim_manifest",
        fail_manifest_write,
    )

    with pytest.raises(RuntimeError, match="manifest write failed"):
        _commit_rebuilt_index(
            claim_path,
            staged_index,
            ClaimManifest.model_validate_json(
                (SAMPLE_OUTPUT / "manifest.json").read_text(encoding="utf-8")
            ),
        )

    assert (current_index / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (current_index / "new.txt").exists()
    assert not list(claim_path.glob(".index-backup-*"))


def _write_fake_index(
    index_path: Path,
    manifest: ClaimManifest,
    *,
    llm_model: str = "provider/model",
) -> None:
    index_path.mkdir(parents=True, exist_ok=True)
    metadata = LightRagIndexMetadata(
        claim_id=manifest.claim_id,
        llm_provider="nvidia",
        llm_model=llm_model,
        embedding_provider=cast(str, manifest.embedding_provider),
        embedding_model=cast(str, manifest.embedding_model),
        embedding_dimension=1024,
        embedding_max_tokens=8192,
        indexed_chunk_count=manifest.chunk_count,
        entity_count=2,
        relationship_count=1,
    )
    (index_path / METADATA_FILE).write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    graph = nx.Graph()
    graph.add_node("Acme", entity_type="ORGANIZATION", description="Insured")
    graph.add_node("Repair Co", entity_type="ORGANIZATION")
    graph.add_edge("Acme", "Repair Co", description="Requested repairs")
    nx.write_graphml(graph, index_path / GRAPH_FILE)
    (index_path / DOC_STATUS_FILE).write_text(
        json.dumps(
            {
                f"source-{index}": {"status": "processed"}
                for index in range(manifest.chunk_count)
            }
        ),
        encoding="utf-8",
    )
