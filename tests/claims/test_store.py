from pathlib import Path
import shutil

from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.claims.models import ClaimSearchResult, DocumentChunk
from knowledge_agent.claims.filesystem import read_jsonl
from knowledge_agent.claims.vector_store import ChromaVectorStore, VectorSearchHit


SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
)


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
    store = ClaimStore(SAMPLE_OUTPUT)

    result = store.search("synthetic collision fnol", top_k=1)[0]
    document = store.get_document("DOC-001")
    page = store.get_page("CLM-SAMPLE-001:p1")

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


def test_knowledge_store_searches_document_metadata():
    store = ClaimStore(SAMPLE_OUTPUT)

    assert store.search("synthetic invoice listing", top_k=1)[0].document_id == (
        "DOC-002"
    )
    assert store.search("invoice", top_k=1)[0].document_type == "invoice"


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
    manifest_path = claim_path / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    semantic = original.replace(
        '"retrieval_mode": "lexical",\n  "embedding_provider": null,\n  "embedding_model": null',
        '"retrieval_mode": "semantic",\n  "embedding_provider": "snowflake",\n  "embedding_model": "test-model"',
    )
    manifest_path.write_text(semantic, encoding="utf-8")
    store = ClaimStore(
        claim_path,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    result = store.search("repair", top_k=1)[0]
    assert result.chunk_id == "DOC-002-CHUNK-001"
    assert result.source_ref.endswith("#DOC-002-CHUNK-001")
    assert result.page_ids == ["CLM-SAMPLE-001:p2"]


def test_semantic_store_queries_a_real_chroma_index(tmp_path):
    claim_path = tmp_path / "claim"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    manifest_path = claim_path / "manifest.json"
    semantic = manifest_path.read_text(encoding="utf-8").replace(
        '"retrieval_mode": "lexical",\n  "embedding_provider": null,\n  "embedding_model": null',
        '"retrieval_mode": "semantic",\n  "embedding_provider": "snowflake",\n  "embedding_model": "test-model"',
    )
    manifest_path.write_text(semantic, encoding="utf-8")

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
        store = ClaimStore(
            claim_path,
            embedder=FakeEmbedder(),
            vector_store=vector_store,
        )
        result = store.search("repair", top_k=1)[0]
    finally:
        vector_store.close()

    assert result.chunk_id == "DOC-002-CHUNK-001"
    assert result.source_ref == "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
