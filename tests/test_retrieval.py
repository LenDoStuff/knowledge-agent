import pytest

from claim_kb import retrieval
from claim_kb.exceptions import ChunkNotFoundError, DocumentNotFoundError
from claim_kb.schemas import (
    DocumentChunk,
    PageRange,
)
from claim_kb.storage import ensure_claim_dirs, write_jsonl


def test_retrieval_functions_use_persisted_metadata_and_mocked_vector_search(
    monkeypatch,
):
    calls = []

    class FakeApi:
        def list_claim_documents(self, claim_id):
            calls.append(("list", claim_id))
            return []

        def search_claim_file(self, claim_id, query, document_types=None, top_k=10):
            calls.append(("search", claim_id, query, document_types, top_k))
            return []

        def read_document_chunk(self, claim_id, document_id, chunk_id):
            calls.append(("read", claim_id, document_id, chunk_id))
            return "chunk"

    monkeypatch.setattr(retrieval, "ClaimKbApi", FakeApi)

    documents = retrieval.list_claim_documents("CLM-001")
    results = retrieval.search_claim_file(
        "CLM-001",
        "loss notice",
        document_types=["fnol"],
        top_k=3,
    )
    chunk = retrieval.read_document_chunk(
        "CLM-001",
        "DOC-001",
        "DOC-001-CHUNK-001",
    )

    assert documents == []
    assert results == []
    assert chunk == "chunk"
    assert calls == [
        ("list", "CLM-001"),
        ("search", "CLM-001", "loss notice", ["fnol"], 3),
        ("read", "CLM-001", "DOC-001", "DOC-001-CHUNK-001"),
    ]


def test_read_document_chunk_errors(tmp_path, monkeypatch):
    data_root = tmp_path / "claims"
    root = ensure_claim_dirs(data_root, "CLM-001")
    write_jsonl(
        root / "chunks" / "chunks.jsonl",
        [
            DocumentChunk(
                claim_id="CLM-001",
                document_id="DOC-001",
                chunk_id="DOC-001-CHUNK-001",
                chunk_index=0,
                document_type="fnol",
                page_range=PageRange(start_page=1, end_page=1),
                text="First notice of loss",
                embedding=[0.1],
            ).model_dump(mode="json")
        ],
    )
    monkeypatch.setenv("CLAIM_KB_DATA_ROOT", str(data_root))

    with pytest.raises(DocumentNotFoundError):
        retrieval.read_document_chunk("CLM-001", "DOC-999", "missing")

    with pytest.raises(ChunkNotFoundError):
        retrieval.read_document_chunk("CLM-001", "DOC-001", "missing")
