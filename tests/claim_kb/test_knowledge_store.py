from pathlib import Path

from claim_kb.knowledge_store import ClaimKbKnowledgeStore
from claim_kb.schemas import KnowledgeItem
from claim_kb.filesystem import read_jsonl


SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claim_kb" / "sample_output"
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
    store = ClaimKbKnowledgeStore(SAMPLE_OUTPUT)

    result = store.search("synthetic collision fnol", top_k=1)[0]
    document = store.get_document("DOC-001")
    page = store.get_page("CLM-SAMPLE-001:p1")

    assert isinstance(result, KnowledgeItem)
    assert result.item_id == "DOC-001-CHUNK-001"
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
    store = ClaimKbKnowledgeStore(SAMPLE_OUTPUT)

    assert store.search("synthetic invoice listing", top_k=1)[0].document_id == (
        "DOC-002"
    )
    assert store.search("invoice", top_k=1)[0].document_type == "invoice"
