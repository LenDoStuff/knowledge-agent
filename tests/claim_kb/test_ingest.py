from pathlib import Path

import pytest
from pypdf import PdfReader

from claim_kb.config import ClaimKbSettings
from claim_kb.classify import DocumentClassification
from claim_kb.ingest import (
    IngestionServices,
    ingest_claim_folder_with_services,
    ingest_claim_pdf_with_services,
)
from claim_kb.schemas import (
    ChunkSearchResult,
    DocumentMetadata,
    PageText,
    PageBoundaryDecision,
    PageRange,
)
from claim_kb.filesystem import read_json, read_jsonl


class FakeOcrClient:
    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        return [
            PageText(
                claim_id=claim_id,
                page_number=1,
                page_id=f"{claim_id}:p1",
                text="First notice of loss for Alice Example.",
                word_count=7,
            ),
            PageText(
                claim_id=claim_id,
                page_number=2,
                page_id=f"{claim_id}:p2",
                text="The same loss notice continues with damage details.",
                word_count=8,
            ),
            PageText(
                claim_id=claim_id,
                page_number=3,
                page_id=f"{claim_id}:p3",
                text="Repair invoice from Contoso Garage.",
                word_count=5,
            ),
            PageText(
                claim_id=claim_id,
                page_number=4,
                page_id=f"{claim_id}:p4",
                text="Invoice line items and total amount.",
                word_count=6,
            ),
        ]


class FakeClassifier:
    def __init__(self) -> None:
        self.boundary_calls = []

    def classify_page_boundary(self, page, prior_page, current_document):
        self.boundary_calls.append((page.page_number, prior_page, current_document))
        if page.page_number == 1:
            return PageBoundaryDecision(
                page_number=1,
                is_new_document=True,
                document_type="fnol",
                title="First Notice of Loss",
                reason="First page",
                confidence=1.0,
            )
        if page.page_number == 3:
            return PageBoundaryDecision(
                page_number=3,
                is_new_document=True,
                document_type="invoice",
                title="Repair Invoice",
                reason="Invoice heading",
                confidence=0.95,
            )
        return PageBoundaryDecision(
            page_number=page.page_number,
            is_new_document=False,
            document_type=current_document.document_type,
            title=current_document.title,
            reason="Continuation of prior page",
            confidence=0.9,
        )

    def extract_document_metadata(self, document, chunks):
        parties = (
            [{"name": "Alice Example", "role": "insured"}]
            if document.document_type == "fnol"
            else [{"name": "Contoso Garage", "role": "repair vendor"}]
        )
        events = (
            [
                {
                    "year": None,
                    "month": None,
                    "day": None,
                    "sentence": "Alice Example reported damage details.",
                    "source_ref": chunks[0].source_ref,
                }
            ]
            if document.document_type == "fnol"
            else [
                {
                    "year": None,
                    "month": None,
                    "day": None,
                    "sentence": "Contoso Garage listed invoice line items.",
                    "source_ref": chunks[0].source_ref,
                }
            ]
        )
        return DocumentMetadata(
            id=document.id,
            title=document.title,
            summary=f"Summary for {document.title}",
            involved_parties=parties,
            events=events,
            document_type=document.document_type,
            page_range=document.page_range,
            file_name=document.file_name,
        )


class FolderOcrClient:
    def __init__(self, page_counts=None) -> None:
        self.page_counts = page_counts or {}
        self.calls = []

    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        self.calls.append(pdf_path.name)
        return [
            PageText(
                claim_id=claim_id,
                page_number=page_number,
                page_id=f"{claim_id}:p{page_number}",
                text=f"{pdf_path.name} page {page_number}",
                word_count=3,
            )
            for page_number in range(
                1,
                self.page_counts.get(pdf_path.name, 1) + 1,
            )
        ]


class FolderClassifier:
    def __init__(self, document_types) -> None:
        self.document_types = document_types
        self.classification_calls = []
        self.boundary_calls = []

    def classify_document(self, file_name, pages):
        self.classification_calls.append((file_name, [page.text for page in pages]))
        document_type = self.document_types[file_name]
        return DocumentClassification(
            title=f"Title for {file_name}",
            document_type=document_type,
        )

    def classify_page_boundary(self, page, prior_page, current_document):
        self.boundary_calls.append((page, prior_page, current_document))
        raise AssertionError("folder ingestion must not classify page boundaries")

    def extract_document_metadata(self, document, chunks):
        return DocumentMetadata(
            id=document.id,
            title=document.title,
            summary=f"Summary for {document.file_name}",
            involved_parties=[],
            events=[
                {
                    "sentence": f"Event in {document.file_name}",
                    "source_ref": chunks[0].source_ref,
                }
            ],
            document_type=document.document_type,
            page_range=document.page_range,
            file_name=document.file_name,
        )


class ConflictingFolderClassifier(FolderClassifier):
    def extract_document_metadata(self, document, chunks):
        metadata = super().extract_document_metadata(document, chunks)
        return metadata.model_copy(update={"document_type": "policy"})


class FakeEmbedder:
    embedding_provider = "snowflake"
    embedding_model = "fake-snowflake-model"

    def embed_texts(self, texts):
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]

    def close(self):
        pass


class FakeVectorStore:
    def __init__(self) -> None:
        self.indexed_chunks = []

    def index_chunks(self, chunks):
        self.indexed_chunks = list(chunks)

    def search(self, query_embedding, document_types, top_k):
        return [
            ChunkSearchResult(
                document_id="DOC-001",
                chunk_id="DOC-001-CHUNK-001",
                page_range=PageRange(start_page=1, end_page=2),
                text="matched text",
                score=0.75,
                document_type="fnol",
            )
        ]

    def close(self):
        pass

    def close(self):
        pass


def test_ingestion_pipeline_creates_expected_outputs(tmp_path, sample_pdf):
    settings = ClaimKbSettings(
        data_root=tmp_path / "claims",
        document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
        snowflake_connection_name="default",
        snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
    )
    classifier = FakeClassifier()
    vector_store = FakeVectorStore()
    services_closed = []
    services = IngestionServices(
        ocr_client=FakeOcrClient(),
        classifier=classifier,
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: vector_store,
        close=lambda: services_closed.append(True),
    )

    claim_file = ingest_claim_pdf_with_services(
        claim_id="CLM-001",
        pdf_path=sample_pdf,
        data_root=settings.data_root,
        services=services,
    )

    root = tmp_path / "claims" / "CLM-001"
    assert (root / "source" / "claim.pdf").exists()
    assert claim_file.source_files == [str(root / "source" / "claim.pdf")]
    assert (root / "documents" / "DOC-001_fnol.pdf").exists()
    assert (root / "documents" / "DOC-002_invoice.pdf").exists()
    assert len(PdfReader(str(root / "documents" / "DOC-001_fnol.pdf")).pages) == 2
    assert len(PdfReader(str(root / "documents" / "DOC-002_invoice.pdf")).pages) == 2

    inventory = claim_file.documents
    assert [document.id for document in inventory] == ["DOC-001", "DOC-002"]
    assert inventory[0].page_range == PageRange(start_page=1, end_page=2)
    assert inventory[0].involved_parties[0].role == "insured"
    assert inventory[0].events[0].year is None
    assert inventory[0].events[0].month is None
    assert inventory[0].events[0].day is None
    assert inventory[0].events[0].source_ref == (
        "CLM-001/DOC-001#DOC-001-CHUNK-001"
    )
    assert inventory[1].document_type == "invoice"

    ocr_rows = read_jsonl(root / "pages.jsonl")
    chunk_rows = read_jsonl(root / "chunks.jsonl")
    assert len(ocr_rows) == 4
    assert ocr_rows[0]["page_id"] == "CLM-001:p1"
    assert len(chunk_rows) == 2
    assert chunk_rows[0]["embedding"]
    assert chunk_rows[0]["document_id"] == "DOC-001"
    assert chunk_rows[0]["page_range"] == {"start_page": 1, "end_page": 2}
    assert chunk_rows[0]["page_ids"] == ["CLM-001:p1", "CLM-001:p2"]
    assert chunk_rows[0]["source_ref"] == (
        "CLM-001/DOC-001#DOC-001-CHUNK-001"
    )
    assert vector_store.indexed_chunks[0].chunk_id == "DOC-001-CHUNK-001"

    page_2_call = classifier.boundary_calls[1]
    assert page_2_call[1].page_number == 1
    assert page_2_call[2].id == "DOC-001"

    assert (root / "manifest.json").exists()
    assert claim_file.embedding_provider == "snowflake"
    assert claim_file.embedding_model == "fake-snowflake-model"
    assert not (root / "metadata").exists()
    assert (root / "run_log.json").exists()
    assert claim_file.chunk_count == 2
    assert services_closed == [True]


def test_keyword_only_ingestion_skips_embeddings_and_clears_index(
    tmp_path,
    sample_pdf,
):
    data_root = tmp_path / "claims"
    stale_index = data_root / "CLM-KEYWORD" / "index" / "chroma"
    stale_index.mkdir(parents=True)
    (stale_index / "stale.bin").write_bytes(b"stale")
    services_closed = []
    services = IngestionServices(
        ocr_client=FakeOcrClient(),
        classifier=FakeClassifier(),
        embedder=None,
        vector_store_factory=None,
        embedding_mode="none",
        close=lambda: services_closed.append(True),
    )

    claim_file = ingest_claim_pdf_with_services(
        claim_id="CLM-KEYWORD",
        pdf_path=sample_pdf,
        data_root=data_root,
        services=services,
    )

    root = data_root / "CLM-KEYWORD"
    chunks = read_jsonl(root / "chunks.jsonl")
    log = read_json(root / "run_log.json")
    assert all(chunk["embedding"] == [] for chunk in chunks)
    assert claim_file.embedding_mode == "none"
    assert claim_file.embedding_provider is None
    assert claim_file.embedding_model is None
    assert claim_file.vector_store_path is None
    assert not (root / "index").exists()
    assert [entry["step"] for entry in log["entries"]][-3:] == [
        "persist_chunks",
        "clear_vector_index",
        "claim_metadata",
    ]
    assert services_closed == [True]


def test_folder_ingestion_ocr_classifies_sorts_and_preserves_pdfs(tmp_path):
    input_path = tmp_path / "input"
    input_path.mkdir()
    payloads = {
        "z_invoice.pdf": b"invoice-z",
        "a_fnol.pdf": b"fnol-a",
        "b_invoice.pdf": b"invoice-b",
    }
    for file_name, payload in payloads.items():
        (input_path / file_name).write_bytes(payload)
    (input_path / "notes.txt").write_text("ignored", encoding="utf-8")
    nested = input_path / "nested"
    nested.mkdir()
    (nested / "nested.pdf").write_bytes(b"ignored")

    ocr_client = FolderOcrClient({"a_fnol.pdf": 2})
    classifier = FolderClassifier(
        {
            "z_invoice.pdf": "invoice",
            "a_fnol.pdf": "fnol",
            "b_invoice.pdf": "invoice",
        }
    )
    vector_store = FakeVectorStore()
    services_closed = []
    services = IngestionServices(
        ocr_client=ocr_client,
        classifier=classifier,
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: vector_store,
        close=lambda: services_closed.append(True),
    )

    claim_file = ingest_claim_folder_with_services(
        claim_id="CLM-FOLDER",
        folder_path=input_path,
        data_root=tmp_path / "claims",
        services=services,
    )

    root = tmp_path / "claims" / "CLM-FOLDER"
    expected_names = ["a_fnol.pdf", "b_invoice.pdf", "z_invoice.pdf"]
    assert [document.id for document in claim_file.documents] == [
        "DOC-001",
        "DOC-002",
        "DOC-003",
    ]
    assert [document.file_name for document in claim_file.documents] == expected_names
    assert [document.document_type for document in claim_file.documents] == [
        "fnol",
        "invoice",
        "invoice",
    ]
    assert [document.page_range for document in claim_file.documents] == [
        PageRange(start_page=1, end_page=2),
        PageRange(start_page=3, end_page=3),
        PageRange(start_page=4, end_page=4),
    ]
    assert [Path(path).name for path in claim_file.source_files] == expected_names
    assert not (root / "source").exists()
    for file_name, payload in payloads.items():
        assert (root / "documents" / file_name).read_bytes() == payload

    pages = read_jsonl(root / "pages.jsonl")
    chunks = read_jsonl(root / "chunks.jsonl")
    assert [page["page_id"] for page in pages] == [
        "CLM-FOLDER:p1",
        "CLM-FOLDER:p2",
        "CLM-FOLDER:p3",
        "CLM-FOLDER:p4",
    ]
    assert chunks[0]["page_ids"] == ["CLM-FOLDER:p1", "CLM-FOLDER:p2"]
    assert [chunk["source_ref"] for chunk in chunks] == [
        "CLM-FOLDER/DOC-001#DOC-001-CHUNK-001",
        "CLM-FOLDER/DOC-002#DOC-002-CHUNK-001",
        "CLM-FOLDER/DOC-003#DOC-003-CHUNK-001",
    ]
    assert sorted(ocr_client.calls) == sorted(payloads)
    assert len(classifier.classification_calls) == 3
    assert classifier.boundary_calls == []
    assert services_closed == [True]


@pytest.mark.parametrize("input_kind", ["missing", "file", "empty"])
def test_folder_ingestion_rejects_invalid_inputs_and_closes_services(
    tmp_path,
    input_kind,
):
    folder_path = tmp_path / input_kind
    if input_kind == "file":
        folder_path.write_text("not a folder", encoding="utf-8")
    elif input_kind == "empty":
        folder_path.mkdir()
    services_closed = []
    services = IngestionServices(
        ocr_client=FolderOcrClient(),
        classifier=FolderClassifier({}),
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: FakeVectorStore(),
        close=lambda: services_closed.append(True),
    )

    expected_error = {
        "missing": FileNotFoundError,
        "file": NotADirectoryError,
        "empty": ValueError,
    }[input_kind]
    with pytest.raises(expected_error):
        ingest_claim_folder_with_services(
            claim_id="CLM-INVALID",
            folder_path=folder_path,
            data_root=tmp_path / "claims",
            services=services,
        )

    assert services_closed == [True]


def test_folder_ingestion_rejects_document_type_changes(tmp_path):
    folder_path = tmp_path / "input"
    folder_path.mkdir()
    (folder_path / "invoice.pdf").write_bytes(b"invoice")
    services_closed = []
    services = IngestionServices(
        ocr_client=FolderOcrClient(),
        classifier=ConflictingFolderClassifier({"invoice.pdf": "invoice"}),
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: FakeVectorStore(),
        close=lambda: services_closed.append(True),
    )

    with pytest.raises(ValueError, match="Document type changed after sorting"):
        ingest_claim_folder_with_services(
            claim_id="CLM-CONFLICT",
            folder_path=folder_path,
            data_root=tmp_path / "claims",
            services=services,
        )

    assert services_closed == [True]
