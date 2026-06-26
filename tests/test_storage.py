from claim_kb.schemas import DocumentChunk, PageRange
from claim_kb.storage import ChromaVectorStore


def test_chroma_vector_store_indexes_searches_and_filters(tmp_path):
    store = ChromaVectorStore("CLM-001", tmp_path / "chroma")
    try:
        store.index_chunks(
            [
                DocumentChunk(
                    claim_id="CLM-001",
                    document_id="DOC-001",
                    chunk_id="DOC-001-CHUNK-001",
                    chunk_index=0,
                    document_type="fnol",
                    page_range=PageRange(start_page=1, end_page=1),
                    text="first notice of loss",
                    embedding=[1.0, 0.0],
                ),
                DocumentChunk(
                    claim_id="CLM-001",
                    document_id="DOC-002",
                    chunk_id="DOC-002-CHUNK-001",
                    chunk_index=1,
                    document_type="invoice",
                    page_range=PageRange(start_page=2, end_page=2),
                    text="repair invoice",
                    embedding=[0.0, 1.0],
                ),
            ]
        )

        results = store.search([1.0, 0.0], document_types=["fnol"], top_k=1)
    finally:
        store.close()

    assert len(results) == 1
    assert results[0].document_id == "DOC-001"
    assert results[0].chunk_id == "DOC-001-CHUNK-001"
    assert results[0].document_type == "fnol"


def test_chroma_vector_store_resets_collection_before_reindex(tmp_path):
    store = ChromaVectorStore("CLM-001", tmp_path / "chroma")
    try:
        store.index_chunks(
            [
                DocumentChunk(
                    claim_id="CLM-001",
                    document_id="DOC-001",
                    chunk_id="DOC-001-CHUNK-001",
                    chunk_index=0,
                    document_type="fnol",
                    page_range=PageRange(start_page=1, end_page=1),
                    text="old embedding dimension",
                    embedding=[1.0, 0.0],
                )
            ]
        )
        store.index_chunks(
            [
                DocumentChunk(
                    claim_id="CLM-001",
                    document_id="DOC-002",
                    chunk_id="DOC-002-CHUNK-001",
                    chunk_index=0,
                    document_type="invoice",
                    page_range=PageRange(start_page=2, end_page=2),
                    text="new embedding dimension",
                    embedding=[1.0, 0.0, 0.0],
                )
            ]
        )

        results = store.search([1.0, 0.0, 0.0], document_types=None, top_k=2)
    finally:
        store.close()

    assert len(results) == 1
    assert results[0].document_id == "DOC-002"
