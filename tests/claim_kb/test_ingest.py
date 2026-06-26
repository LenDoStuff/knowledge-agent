from pathlib import Path

from pypdf import PdfReader

from claim_kb.config import ClaimKbSettings
from claim_kb.ingest import IngestionServices, ingest_claim_pdf_with_services
from claim_kb.schemas import (
    ChunkSearchResult,
    DocumentMetadata,
    PageText,
    PageBoundaryDecision,
    PageRange,
)
from claim_kb.storage import read_jsonl


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


def test_ingestion_pipeline_creates_expected_outputs(tmp_path, sample_pdf):
    settings = ClaimKbSettings(
        data_root=tmp_path / "claims",
        ai_project_endpoint="https://example.services.ai.azure.com/api/projects/proj",
        document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
        openai_deployment="gpt-test",
        tenant_id=None,
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
    )

    claim_file = ingest_claim_pdf_with_services(
        claim_id="CLM-001",
        pdf_path=sample_pdf,
        settings=settings,
        services=services,
    )

    root = tmp_path / "claims" / "CLM-001"
    assert (root / "source" / "claim.pdf").exists()
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
