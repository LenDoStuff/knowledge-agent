from pathlib import Path

import pytest
from pypdf import PdfReader

from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.classify import DocumentClassification, PageBoundaryDecision
from knowledge_agent.claims.pipeline import (
    IngestionServices,
    ingest_claim_folder,
    ingest_claim_pdf,
)
from knowledge_agent.claims.models import (
    DocumentMetadata,
    PageText,
    PageRange,
)
from knowledge_agent.claims.filesystem import read_json, read_jsonl


class FakeOcrClient:
    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        return [
            PageText(
                claim_id=claim_id,
                page_number=1,
                page_id=f"{claim_id}:p1",
                text="First notice of loss for Alice Example.",
            ),
            PageText(
                claim_id=claim_id,
                page_number=2,
                page_id=f"{claim_id}:p2",
                text="The same loss notice continues with damage details.",
            ),
            PageText(
                claim_id=claim_id,
                page_number=3,
                page_id=f"{claim_id}:p3",
                text="Repair invoice from Contoso Garage.",
            ),
            PageText(
                claim_id=claim_id,
                page_number=4,
                page_id=f"{claim_id}:p4",
                text="Invoice line items and total amount.",
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
            )
        if page.page_number == 3:
            return PageBoundaryDecision(
                page_number=3,
                is_new_document=True,
                document_type="invoice",
                title="Repair Invoice",
            )
        return PageBoundaryDecision(
            page_number=page.page_number,
            is_new_document=False,
            document_type=current_document.document_type,
            title=current_document.title,
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
        self.indexed_embeddings = []

    def index_chunks(self, chunks, embeddings):
        self.indexed_chunks = list(chunks)
        self.indexed_embeddings = list(embeddings)

    def close(self):
        pass


def test_ingestion_pipeline_creates_expected_outputs(tmp_path, sample_pdf):
    settings = ClaimSettings(
        data_root=tmp_path / "claims",
        document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
        snowflake_connection_name="default",
        snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
    )
    classifier = FakeClassifier()
    vector_store = FakeVectorStore()
    services = IngestionServices(
        ocr_client=FakeOcrClient(),
        classifier=classifier,
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: vector_store,
        retrieval_mode="semantic",
    )

    claim_file = ingest_claim_pdf(
        claim_id="CLM-001",
        pdf_path=sample_pdf,
        data_root=settings.data_root,
        services=services,
    )

    root = tmp_path / "claims" / "CLM-001"
    assert (root / "source" / "claim.pdf").exists()
    assert claim_file.source_files == ["source/claim.pdf"]
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
    assert "embedding" not in chunk_rows[0]
    assert chunk_rows[0]["document_id"] == "DOC-001"
    assert chunk_rows[0]["page_range"] == {"start_page": 1, "end_page": 2}
    assert chunk_rows[0]["page_ids"] == ["CLM-001:p1", "CLM-001:p2"]
    assert chunk_rows[0]["source_ref"] == (
        "CLM-001/DOC-001#DOC-001-CHUNK-001"
    )
    assert vector_store.indexed_chunks[0].chunk_id == "DOC-001-CHUNK-001"
    assert vector_store.indexed_embeddings[0]

    page_2_call = classifier.boundary_calls[1]
    assert page_2_call[1].page_number == 1
    assert page_2_call[2].id == "DOC-001"

    assert (root / "manifest.json").exists()
    assert claim_file.embedding_provider == "snowflake"
    assert claim_file.embedding_model == "fake-snowflake-model"
    assert claim_file.retrieval_mode == "semantic"
    assert not (root / "metadata").exists()
    assert (root / "run_log.json").exists()
    assert claim_file.chunk_count == 2


def test_lexical_ingestion_skips_embeddings_and_clears_index(
    tmp_path,
    sample_pdf,
):
    data_root = tmp_path / "claims"
    stale_index = data_root / "CLM-KEYWORD" / "index" / "chroma"
    stale_index.mkdir(parents=True)
    (stale_index / "stale.bin").write_bytes(b"stale")
    services = IngestionServices(
        ocr_client=FakeOcrClient(),
        classifier=FakeClassifier(),
        embedder=None,
        vector_store_factory=None,
        retrieval_mode="lexical",
    )

    claim_file = ingest_claim_pdf(
        claim_id="CLM-KEYWORD",
        pdf_path=sample_pdf,
        data_root=data_root,
        services=services,
    )

    root = data_root / "CLM-KEYWORD"
    chunks = read_jsonl(root / "chunks.jsonl")
    log = read_json(root / "run_log.json")
    assert all("embedding" not in chunk for chunk in chunks)
    assert claim_file.retrieval_mode == "lexical"
    assert claim_file.embedding_provider is None
    assert claim_file.embedding_model is None
    assert not (root / "index").exists()
    assert [entry["step"] for entry in log["entries"]][-3:] == [
        "persist_chunks",
        "clear_vector_index",
        "claim_manifest",
    ]


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
    services = IngestionServices(
        ocr_client=ocr_client,
        classifier=classifier,
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: vector_store,
        retrieval_mode="semantic",
    )

    claim_file = ingest_claim_folder(
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


@pytest.mark.parametrize("input_kind", ["missing", "file", "empty"])
def test_folder_ingestion_rejects_invalid_inputs(
    tmp_path,
    input_kind,
):
    folder_path = tmp_path / input_kind
    if input_kind == "file":
        folder_path.write_text("not a folder", encoding="utf-8")
    elif input_kind == "empty":
        folder_path.mkdir()
    services = IngestionServices(
        ocr_client=FolderOcrClient(),
        classifier=FolderClassifier({}),
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: FakeVectorStore(),
        retrieval_mode="semantic",
    )

    expected_error = {
        "missing": FileNotFoundError,
        "file": NotADirectoryError,
        "empty": ValueError,
    }[input_kind]
    with pytest.raises(expected_error):
        ingest_claim_folder(
            claim_id="CLM-INVALID",
            folder_path=folder_path,
            data_root=tmp_path / "claims",
            services=services,
        )



def test_folder_ingestion_rejects_document_type_changes(tmp_path):
    folder_path = tmp_path / "input"
    folder_path.mkdir()
    (folder_path / "invoice.pdf").write_bytes(b"invoice")
    services = IngestionServices(
        ocr_client=FolderOcrClient(),
        classifier=ConflictingFolderClassifier({"invoice.pdf": "invoice"}),
        embedder=FakeEmbedder(),
        vector_store_factory=lambda root: FakeVectorStore(),
        retrieval_mode="semantic",
    )

    with pytest.raises(ValueError, match="Document type changed after sorting"):
        ingest_claim_folder(
            claim_id="CLM-CONFLICT",
            folder_path=folder_path,
            data_root=tmp_path / "claims",
            services=services,
        )
