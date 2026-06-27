import pytest

from claim_kb import (
    ClaimKbApi,
    DocumentEvent,
    DocumentParty,
    ingest_claim_folder,
    search_claim_file,
)
from claim_kb.config import ClaimKbSettings
from claim_kb.exceptions import ChunkNotFoundError, DocumentNotFoundError
from claim_kb.schemas import (
    ChunkSearchResult,
    DocumentChunk,
    DocumentMetadata,
    PageRange,
    PageText,
    StructuredClaimFile,
)
from claim_kb.filesystem import (
    ensure_claim_dirs,
    write_claim_metadata,
    write_jsonl,
)


class FakeEmbedder:
    def __init__(self, model_log):
        self.model_log = model_log

    def embed_texts(self, texts):
        assert texts == ["invoice total"]
        return [[0.4, 0.5]]

    def close(self):
        pass


class ClosingVectorStore:
    def __init__(self) -> None:
        self.closed = False
        self.calls = []

    def search(self, query_embedding, document_types, top_k):
        self.calls.append((query_embedding, document_types, top_k))
        return [
            ChunkSearchResult(
                document_id="DOC-002",
                chunk_id="DOC-002-CHUNK-001",
                page_range=PageRange(start_page=2, end_page=2),
                text="Invoice total is present.",
                score=0.8,
                document_type="invoice",
            )
        ]

    def index_chunks(self, chunks):
        raise AssertionError("search test should not index")

    def close(self):
        self.closed = True


def test_claim_kb_api_supports_internal_module_usage(tmp_path):
    data_root = tmp_path / "claims"
    root = ensure_claim_dirs(data_root, "CLM-001")
    documents = [
        DocumentMetadata(
            id="DOC-002",
            title="Repair Invoice",
            summary="Invoice summary",
            involved_parties=[
                {"name": "Contoso Garage", "role": "repair vendor"},
            ],
            events=[
                {
                    "year": 2026,
                    "month": 6,
                    "day": None,
                    "sentence": "Contoso Garage issued a repair invoice.",
                    "source_ref": "CLM-001/DOC-002#DOC-002-CHUNK-001",
                },
            ],
            document_type="invoice",
            page_range=PageRange(start_page=2, end_page=2),
            file_name="DOC-002_invoice.pdf",
        )
    ]
    write_jsonl(
        root / "chunks.jsonl",
        [
            DocumentChunk(
                claim_id="CLM-001",
                document_id="DOC-002",
                chunk_id="DOC-002-CHUNK-001",
                source_ref="CLM-001/DOC-002#DOC-002-CHUNK-001",
                chunk_index=0,
                document_type="invoice",
                page_range=PageRange(start_page=2, end_page=2),
                page_ids=["CLM-001:p2"],
                text="Invoice total is present.",
                embedding=[0.4, 0.5],
            ).model_dump(mode="json")
        ],
    )
    write_claim_metadata(
        root,
        StructuredClaimFile(
            claim_id="CLM-001",
            root_path=str(root),
            source_files=[str(root / "source" / "claim.pdf")],
            documents=documents,
            chunk_count=1,
            vector_store_path=str(root / "index" / "chroma"),
            embedding_provider="snowflake",
            embedding_model="stored-snowflake-model",
        ),
    )
    settings = ClaimKbSettings(
        data_root=data_root,
        document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
        snowflake_connection_name="default",
        snowflake_embedding_model="configured-snowflake-model",
    )
    vector_store = ClosingVectorStore()
    model_log = []

    def build_embedder(settings):
        model_log.append(settings.snowflake_embedding_model)
        return FakeEmbedder(model_log)

    api = ClaimKbApi(
        settings=settings,
        embedder_factory=build_embedder,
        vector_store_factory=lambda settings, claim_id: vector_store,
    )

    documents = api.list_claim_documents("CLM-001")
    results = api.search_claim_file(
        "CLM-001",
        "invoice total",
        document_types=["invoice"],
        top_k=5,
    )
    chunk = api.read_document_chunk(
        "CLM-001",
        "DOC-002",
        "DOC-002-CHUNK-001",
    )

    assert documents[0].title == "Repair Invoice"
    assert documents[0].involved_parties[0].role == "repair vendor"
    assert documents[0].events[0].year == 2026
    assert documents[0].events[0].month == 6
    assert documents[0].events[0].day is None
    assert results[0].document_id == "DOC-002"
    assert chunk.chunk_id == "DOC-002-CHUNK-001"
    assert model_log == ["stored-snowflake-model"]
    assert vector_store.calls == [([0.4, 0.5], ["invoice"], 5)]
    assert vector_store.closed


def test_claim_kb_api_searches_keyword_only_claim_without_vector_services(tmp_path):
    data_root = tmp_path / "claims"
    root = ensure_claim_dirs(data_root, "CLM-KEYWORD")
    documents = [
        DocumentMetadata(
            id="DOC-001",
            title="Repair Invoice",
            summary="Garage bill",
            document_type="invoice",
            page_range=PageRange(start_page=1, end_page=1),
            file_name="invoice.pdf",
        ),
        DocumentMetadata(
            id="DOC-002",
            title="First Notice",
            summary="Initial claim",
            document_type="fnol",
            page_range=PageRange(start_page=2, end_page=2),
            file_name="fnol.pdf",
        ),
    ]
    pages = [
        PageText(
            claim_id="CLM-KEYWORD",
            page_number=1,
            page_id="CLM-KEYWORD:p1",
            text="Repair total 850 shared",
        ),
        PageText(
            claim_id="CLM-KEYWORD",
            page_number=2,
            page_id="CLM-KEYWORD:p2",
            text="Repair was reported shared",
        ),
    ]
    chunks = [
        DocumentChunk(
            claim_id="CLM-KEYWORD",
            document_id="DOC-001",
            chunk_id="DOC-001-CHUNK-001",
            source_ref="CLM-KEYWORD/DOC-001#DOC-001-CHUNK-001",
            chunk_index=0,
            document_type="invoice",
            page_range=PageRange(start_page=1, end_page=1),
            page_ids=["CLM-KEYWORD:p1"],
            text="Repair total 850 shared",
        ),
        DocumentChunk(
            claim_id="CLM-KEYWORD",
            document_id="DOC-002",
            chunk_id="DOC-002-CHUNK-001",
            source_ref="CLM-KEYWORD/DOC-002#DOC-002-CHUNK-001",
            chunk_index=1,
            document_type="fnol",
            page_range=PageRange(start_page=2, end_page=2),
            page_ids=["CLM-KEYWORD:p2"],
            text="Repair was reported shared",
        ),
    ]
    write_jsonl(
        root / "pages.jsonl",
        [page.model_dump(mode="json") for page in pages],
    )
    write_jsonl(
        root / "chunks.jsonl",
        [chunk.model_dump(mode="json") for chunk in chunks],
    )
    write_claim_metadata(
        root,
        StructuredClaimFile(
            claim_id="CLM-KEYWORD",
            root_path=str(root),
            source_files=[str(root / "invoice.pdf")],
            documents=documents,
            chunk_count=2,
            vector_store_path=None,
            embedding_provider=None,
            embedding_model=None,
            embedding_mode="none",
        ),
    )
    settings = ClaimKbSettings(
        data_root=data_root,
        document_intelligence_endpoint=None,
        snowflake_connection_name="",
        snowflake_embedding_model="",
    )

    def unexpected_factory(*args):
        raise AssertionError("keyword search must not construct vector services")

    api = ClaimKbApi(
        settings=settings,
        embedder_factory=unexpected_factory,
        vector_store_factory=unexpected_factory,
    )

    results = api.search_claim_file(
        "CLM-KEYWORD",
        "repair total",
        document_types=["invoice"],
        top_k=5,
    )

    assert [result.chunk_id for result in results] == ["DOC-001-CHUNK-001"]
    assert results[0].score == 3.0
    assert api.search_claim_file("CLM-KEYWORD", "repair", top_k=1)[
        0
    ].document_type == "invoice"
    assert [
        result.chunk_id
        for result in api.search_claim_file("CLM-KEYWORD", "shared", top_k=2)
    ] == ["DOC-001-CHUNK-001", "DOC-002-CHUNK-001"]
    assert api.search_claim_file(
        "CLM-KEYWORD",
        "repair",
        document_types=["policy"],
    ) == []
    with pytest.raises(ValueError, match="searchable text"):
        api.search_claim_file("CLM-KEYWORD", "  ")
    with pytest.raises(ValueError, match="top_k"):
        api.search_claim_file("CLM-KEYWORD", "repair", top_k=0)


def test_package_level_imports_expose_api_facade():
    assert ClaimKbApi.__name__ == "ClaimKbApi"
    assert DocumentEvent.__name__ == "DocumentEvent"
    assert DocumentParty.__name__ == "DocumentParty"
    assert callable(ingest_claim_folder)
    assert callable(search_claim_file)


def test_ingestion_api_uses_mode_derived_services(tmp_path):
    calls = []

    def record_factory(claim_id, settings):
        calls.append(claim_id)
        raise RuntimeError("factory called")

    api = ClaimKbApi(
        settings=ClaimKbSettings(
            data_root=tmp_path,
            document_intelligence_endpoint=None,
            snowflake_connection_name="default",
            snowflake_embedding_model="model",
        ),
        ingestion_services_factory=record_factory,
    )

    with pytest.raises(RuntimeError, match="factory called"):
        api.ingest_claim_pdf("CLM-001", "claim.pdf")
    assert calls == ["CLM-001"]


def test_read_document_chunk_errors(tmp_path):
    data_root = tmp_path / "claims"
    root = ensure_claim_dirs(data_root, "CLM-001")
    write_jsonl(
        root / "chunks.jsonl",
        [
            DocumentChunk(
                claim_id="CLM-001",
                document_id="DOC-001",
                chunk_id="DOC-001-CHUNK-001",
                source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
                chunk_index=0,
                document_type="fnol",
                page_range=PageRange(start_page=1, end_page=1),
                page_ids=["CLM-001:p1"],
                text="First notice of loss",
                embedding=[0.1],
            ).model_dump(mode="json")
        ],
    )
    api = ClaimKbApi(
        settings=ClaimKbSettings(
            data_root=data_root,
            document_intelligence_endpoint=None,
            snowflake_connection_name="default",
            snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
        )
    )

    with pytest.raises(DocumentNotFoundError):
        api.read_document_chunk("CLM-001", "DOC-999", "missing")

    with pytest.raises(ChunkNotFoundError):
        api.read_document_chunk("CLM-001", "DOC-001", "missing")
