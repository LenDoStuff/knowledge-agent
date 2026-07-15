"""Tests for persisted claim loading and retrieval."""

import json
import shutil
from pathlib import Path

import pytest

from knowledge_agent.claims.store import (
    get_document,
    get_page,
    load_claim_store,
    search_claim,
)
from knowledge_agent.claims.models import ClaimSearchResult, DocumentChunk
from knowledge_agent.claims.filesystem import read_jsonl
from knowledge_agent.claims.vector_store import ChromaVectorStore, VectorSearchHit


SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
)


def set_retrieval_manifest(
    claim_path: Path,
    mode: str,
    provider: str,
    model: str,
) -> None:
    manifest_path = claim_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        retrieval_mode=mode,
        embedding_provider=provider,
        embedding_model=model,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_sample_output_has_stable_citation_fields():
    pages = read_jsonl(SAMPLE_OUTPUT / "pages.jsonl")
    chunks = read_jsonl(SAMPLE_OUTPUT / "chunks.jsonl")

    assert [page["page_id"] for page in pages] == [
        "CLM-SAMPLE-001:p1",
        "CLM-SAMPLE-001:p2",
    ]
    assert chunks[0]["source_ref"] == (
        "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"
    )
    assert chunks[0]["page_ids"] == ["CLM-SAMPLE-001:p1"]


def test_knowledge_store_loads_searches_and_reads_sample_output():
    store = load_claim_store(SAMPLE_OUTPUT)

    result = search_claim(store, "synthetic collision fnol", top_k=1)[0]
    document = get_document(store, "DOC-001")
    page = get_page(store, "CLM-SAMPLE-001:p1")

    assert isinstance(result, ClaimSearchResult)
    assert result.chunk_id == "DOC-001-CHUNK-001"
    assert result.document_id == "DOC-001"
    assert result.document_type == "fnol"
    assert result.document_title == "First Notice of Loss"
    assert result.document_summary == (
        "Synthetic notice describing a sample collision claim."
    )
    assert result.page_ids == ["CLM-SAMPLE-001:p1"]
    assert result.source_ref == (
        "CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001"
    )
    assert document.events[0].source_ref == result.source_ref
    assert page.page_number == 1
    assert "First Notice of Loss" in page.text


def test_knowledge_store_exposes_ordered_pages_and_chunks_for_inspection():
    store = load_claim_store(SAMPLE_OUTPUT)

    assert [page.page_id for page in store.pages] == [
        "CLM-SAMPLE-001:p1",
        "CLM-SAMPLE-001:p2",
    ]
    assert [chunk.chunk_id for chunk in store.chunks] == [
        "DOC-001-CHUNK-001",
        "DOC-002-CHUNK-001",
    ]


def test_knowledge_store_searches_document_metadata():
    store = load_claim_store(SAMPLE_OUTPUT)

    assert search_claim(store, "synthetic invoice listing", top_k=1)[0].document_id == (
        "DOC-002"
    )
    assert search_claim(store, "invoice", top_k=1)[0].document_type == "invoice"


def test_semantic_store_returns_the_same_citation_rich_result_shape(tmp_path):
    class FakeEmbedder:
        embedding_provider = "snowflake"
        embedding_model = "test-model"

        def embed_texts(self, texts):
            assert texts == ["repair"]
            return [[1.0, 0.0]]

        def close(self):
            pass

    class FakeVectorStore:
        def search(self, query_embedding, document_types, top_k):
            assert query_embedding == [1.0, 0.0]
            return [VectorSearchHit("DOC-002-CHUNK-001", 0.9)]

        def index_chunks(self, chunks, embeddings):
            raise AssertionError("not used")

        def close(self):
            pass

    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    set_retrieval_manifest(claim_path, "semantic", "snowflake", "test-model")
    store = load_claim_store(
        claim_path,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    result = search_claim(store, "repair", top_k=1)[0]
    assert result.chunk_id == "DOC-002-CHUNK-001"
    assert result.source_ref.endswith("#DOC-002-CHUNK-001")
    assert result.page_ids == ["CLM-SAMPLE-001:p2"]


def test_semantic_store_queries_a_real_chroma_index(tmp_path):
    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    set_retrieval_manifest(claim_path, "semantic", "snowflake", "test-model")

    class FakeEmbedder:
        embedding_provider = "snowflake"
        embedding_model = "test-model"

        def embed_texts(self, texts):
            return [[0.0, 1.0]]

        def close(self):
            pass

    chunks = [
        DocumentChunk.model_validate(row)
        for row in read_jsonl(claim_path / "chunks.jsonl")
    ]
    vector_store = ChromaVectorStore("CLM-SAMPLE-001", claim_path / "index/chroma")
    try:
        vector_store.index_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        store = load_claim_store(
            claim_path,
            embedder=FakeEmbedder(),
            vector_store=vector_store,
        )
        result = search_claim(store, "repair", top_k=1)[0]
    finally:
        vector_store.close()

    assert result.chunk_id == "DOC-002-CHUNK-001"
    assert result.source_ref == "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"


def test_lightrag_store_rejects_a_missing_index(tmp_path):
    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    set_retrieval_manifest(claim_path, "lightrag", "nvidia", "baai/bge-m3")

    with pytest.raises(FileNotFoundError, match="metadata"):
        load_claim_store(claim_path)


def test_store_selects_one_of_two_persisted_retrieval_modes(tmp_path):
    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    manifest_path = claim_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        additional_retrieval_modes=["lightrag"],
        embedding_provider="nvidia",
        embedding_model="baai/bge-m3",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    custom = load_claim_store(claim_path, retrieval_mode="lexical")
    assert custom.retrieval_mode == "lexical"
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_claim_store(claim_path, retrieval_mode="lightrag")
    with pytest.raises(ValueError, match="not available"):
        load_claim_store(claim_path, retrieval_mode="semantic")
