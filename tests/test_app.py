"""Tests for Streamlit claim, knowledge-base, and research views."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import pytest
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_core import to_jsonable_python
from streamlit.testing.v1 import AppTest

from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.errors import ChunkNotFoundError
from knowledge_agent.claims.lightrag import (
    DOC_STATUS_FILE,
    GRAPH_FILE,
    METADATA_FILE,
    LightRagIndexMetadata,
)
from knowledge_agent.claims.store import load_claim_store
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.research.history import (
    ClarificationExchange,
    ResearchInteraction,
    load_research_history,
    store_interaction,
)
from knowledge_agent.ui.claims import (
    COMBINED_LABEL,
    SEPARATE_LABEL,
    discover_claims,
    ingest_uploads,
    validate_uploads,
)
from knowledge_agent.ui.knowledge_base import (
    document_rows,
    party_rows,
    timeline_rows,
)
from knowledge_agent.ui.reports import cited_answer_html, claim_search_trace
from knowledge_agent.ui.research import is_greeting


REPO_ROOT = Path(__file__).parents[1]
SAMPLE_OUTPUT = REPO_ROOT / "examples" / "claims" / "sample_output"
MISSING_SOURCE = "CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"


def copy_sample_claim(tmp_path: Path) -> Path:
    data_root = tmp_path / "claims"
    claim_path = data_root / "CLM-SAMPLE-001"
    shutil.copytree(SAMPLE_OUTPUT, claim_path)
    return claim_path


def write_test_lightrag_index(claim_path: Path) -> None:
    index_path = claim_path / "index" / "lightrag"
    index_path.mkdir(parents=True)
    metadata = LightRagIndexMetadata(
        claim_id="CLM-SAMPLE-001",
        llm_provider="nvidia",
        llm_model="provider/model",
        embedding_provider="nvidia",
        embedding_model="baai/bge-m3",
        embedding_dimension=1024,
        embedding_max_tokens=8192,
        indexed_chunk_count=2,
        entity_count=2,
        relationship_count=1,
    )
    (index_path / METADATA_FILE).write_text(
        metadata.model_dump_json(), encoding="utf-8"
    )
    graph = nx.Graph()
    graph.add_node("Acme", entity_type="ORGANIZATION")
    graph.add_node("Repair Co", entity_type="ORGANIZATION")
    graph.add_edge("Acme", "Repair Co", description="repaired")
    nx.write_graphml(graph, index_path / GRAPH_FILE)
    (index_path / DOC_STATUS_FILE).write_text(
        json.dumps(
            {
                f"source-{index}": {"status": "processed"}
                for index in range(2)
            }
        ),
        encoding="utf-8",
    )


class FakeUpload:
    def __init__(self, name: str, payload: bytes = b"pdf") -> None:
        self.name = name
        self.payload = payload

    def getvalue(self) -> bytes:
        return self.payload


def claim_settings(data_root: Path) -> ClaimSettings:
    return ClaimSettings(
        data_root=data_root,
        document_intelligence_endpoint="https://example.cognitiveservices.azure.com",
        document_intelligence_api_key="test-key",
        snowflake_connection_name="default",
        snowflake_embedding_model="snowflake-arctic-embed-l-v2.0",
    )


def llm_settings() -> LlmSettings:
    return LlmSettings(
        profile="api_key",
        model="test-model",
        reasoning_effort="low",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_api_key_ds4="test-key",
    )


def test_discover_claims_returns_valid_entries_and_visible_errors(tmp_path):
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    shutil.copy2(SAMPLE_OUTPUT / "manifest.json", valid_path / "manifest.json")
    (tmp_path / "invalid_claim").mkdir()

    claims, invalid = discover_claims(tmp_path)

    assert [claim.manifest.claim_id for claim in claims] == ["CLM-SAMPLE-001"]
    assert invalid[0].path.name == "invalid_claim"
    assert "manifest.json" in invalid[0].error


@pytest.mark.parametrize(
    ("mode", "uploads", "message"),
    [
        ("separate", [], "at least one PDF"),
        ("combined", [FakeUpload("a.pdf"), FakeUpload("b.pdf")], "exactly one"),
        ("separate", [FakeUpload("notes.txt")], "Only PDF"),
        (
            "separate",
            [FakeUpload("invoice.pdf"), FakeUpload("INVOICE.PDF")],
            "Duplicate",
        ),
    ],
)
def test_validate_uploads_rejects_invalid_inputs(mode, uploads, message):
    with pytest.raises(ValueError, match=message):
        validate_uploads(mode, uploads)


def test_ingest_uploads_rejects_existing_claim_before_live_services(tmp_path):
    (tmp_path / "CLM-001").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ingest_uploads(
            "CLM-001",
            "combined",
            [FakeUpload("claim.pdf")],
            claim_settings(tmp_path),
            llm_settings(),
        )


def test_ingest_uploads_removes_output_created_by_a_failed_run(
    monkeypatch, tmp_path
):
    @contextmanager
    def fake_services(*args):
        yield object()

    def fail_after_creating_output(claim_id, pdf_path, data_root, services):
        output = data_root / claim_id
        output.mkdir()
        (output / "run_log.json").write_text("failed", encoding="utf-8")
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(
        "knowledge_agent.ui.claims.live_ingestion_services",
        fake_services,
    )
    monkeypatch.setattr(
        "knowledge_agent.ui.claims.ingest_claim_pdf",
        fail_after_creating_output,
    )

    with pytest.raises(RuntimeError, match="metadata failed"):
        ingest_uploads(
            "CLM-FAILED",
            "combined",
            [FakeUpload("claim.pdf")],
            claim_settings(tmp_path),
            llm_settings(),
        )

    assert not (tmp_path / "CLM-FAILED").exists()


def test_streamlit_app_renders_without_api_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(tmp_path))
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))

    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Claim Research Workbench"
    assert "Ingest a claim" in app.info[0].value
    assert next(item for item in app.radio if item.label == "Knowledge base").value == (
        "Custom"
    )
    assert next(
        item for item in app.radio if item.label == "Knowledge base"
    ).options == ["Custom", "LightRAG", "Both"]


def test_streamlit_app_renders_the_sample_knowledge_base(monkeypatch):
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(SAMPLE_OUTPUT.parent))
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))

    app.run(timeout=10)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Knowledge base",
        "Timeline",
        "Parties",
        "Documents",
        "Metadata",
        "Evidence",
        "OCR pages",
        "Research chat",
        "New research",
        "Report history",
    ]
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Claim", "CLM-SAMPLE-001"),
        ("Documents", "2"),
        ("Chunks", "2"),
        ("Retrieval", "lexical"),
        ("Selected document", "DOC-001"),
        ("Pages", "1"),
        ("Evidence chunks", "1"),
    ]
    assert [(toggle.label, toggle.value) for toggle in app.toggle] == [
        ("Planning", True),
        ("Show live audit", False),
    ]
    engine = next(
        item for item in app.selectbox if item.label == "Research knowledge base"
    )
    assert engine.value == "lexical"
    assert engine.options == ["Custom (lexical)"]


def test_claim_overview_aggregates_timeline_and_parties():
    store = load_claim_store(SAMPLE_OUTPUT)

    timeline = timeline_rows(store)
    parties = party_rows(store)

    assert timeline[0]["Date"] == "2026-06-01"
    assert timeline[-1]["Date"] == "Undated"
    assert {party["Party"] for party in parties} == {
        "Casey Sample",
        "Example Mutual",
        "Sample Body Shop",
    }


def test_streamlit_renders_lightrag_graph_and_rebuild_controls(monkeypatch, tmp_path):
    claim_path = copy_sample_claim(tmp_path)
    manifest_path = claim_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        retrieval_mode="lightrag",
        embedding_provider="nvidia",
        embedding_model="baai/bge-m3",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_test_lightrag_index(claim_path)
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(claim_path.parent))

    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.run(timeout=10)

    assert not app.exception
    assert any(item.value == "LightRAG graph" for item in app.subheader)
    assert {tab.label for tab in app.tabs}.issuperset({"Entities", "Relationships"})
    assert any(item.label == "Rebuild index" and not item.disabled for item in app.button)
    assert ("Entities", "2") in [(item.label, item.value) for item in app.metric]


def test_streamlit_both_claim_lets_research_choose_an_engine(monkeypatch, tmp_path):
    claim_path = copy_sample_claim(tmp_path)
    manifest_path = claim_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        additional_retrieval_modes=["lightrag"],
        embedding_provider="nvidia",
        embedding_model="baai/bge-m3",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_test_lightrag_index(claim_path)
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(claim_path.parent))

    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.run(timeout=10)

    assert not app.exception
    assert ("Retrieval", "lexical + lightrag") in [
        (item.label, item.value) for item in app.metric
    ]
    research_engine = next(
        item for item in app.selectbox if item.label == "Research knowledge base"
    )
    assert research_engine.options == ["Custom (lexical)", "LightRAG"]
    rebuild_engine = next(
        item for item in app.selectbox if item.label == "Target engine"
    )
    assert rebuild_engine.value == "Both"


def test_streamlit_surfaces_corrupt_lightrag_and_keeps_rebuild_available(
    monkeypatch,
    tmp_path,
):
    claim_path = copy_sample_claim(tmp_path)
    manifest_path = claim_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        retrieval_mode="lightrag",
        embedding_provider="nvidia",
        embedding_model="baai/bge-m3",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_test_lightrag_index(claim_path)
    status_path = claim_path / "index" / "lightrag" / DOC_STATUS_FILE
    statuses = json.loads(status_path.read_text(encoding="utf-8"))
    statuses[next(iter(statuses))]["status"] = "failed"
    status_path.write_text(json.dumps(statuses), encoding="utf-8")
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(claim_path.parent))

    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.run(timeout=10)

    assert not app.exception
    assert any("incomplete documents" in item.value for item in app.error)
    assert not next(
        button for button in app.button if button.label == "Rebuild index"
    ).disabled


def test_document_inventory_summarizes_extracted_metadata():
    rows = document_rows(load_claim_store(SAMPLE_OUTPUT))

    assert rows[0] == {
        "ID": "DOC-001",
        "Type": "fnol",
        "Title": "First Notice of Loss",
        "Pages": "1",
        "Parties": 2,
        "Events": 2,
        "Evidence chunks": 1,
        "File": "DOC-001_fnol.pdf",
    }


def test_streamlit_app_renders_native_audit_trail(monkeypatch, tmp_path):
    claim_path = copy_sample_claim(tmp_path)
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(claim_path.parent))
    agent_messages = [
        ModelRequest(parts=[UserPromptPart("Inspect the repair invoice.")]),
        ModelResponse(parts=[]),
    ]
    store_interaction(
        claim_path,
        ResearchInteraction(
            claim_id="CLM-SAMPLE-001",
            status="completed",
            question="What was repaired?",
            planning_enabled=True,
            output={
                "answer": "The bumper was repaired.",
                "source_refs": [],
                "evidence_sufficient": False,
            },
            agent_messages=agent_messages,
            audit_events=[
                {"type": "FunctionToolCallEvent", "payload": {"part": {}}}
            ],
        ),
    )
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))

    app.run(timeout=10)

    assert not app.exception
    assert any(
        "Saved reports and audits contain conversation text" in item.value
        for item in app.warning
    )
    assert any(
        item.label == "Audit trail (2 messages · 1 events)"
        for item in app.expander
    )
    assert any(
        item.value == "Knowledge base: engine not recorded" for item in app.caption
    )


def test_pending_plan_resumes_and_can_be_cancelled(monkeypatch, tmp_path):
    claim_path = copy_sample_claim(tmp_path)
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(claim_path.parent))
    item = ResearchInteraction(
        claim_id="CLM-SAMPLE-001",
        status="awaiting_approval",
        question="What repairs were invoiced?",
        planning_enabled=True,
        clarifications=[
            ClarificationExchange(
                question="Which dates are in scope?",
                reason="The period was ambiguous.",
                answer="All dates.",
            )
        ],
        plan={
            "objective": "Identify invoiced repairs.",
            "understood_scope": "All repair invoices and dates.",
            "assumptions": [],
            "searches": [
                {
                    "query": "repair invoice",
                    "research_goal": "Find repaired items.",
                }
            ],
            "completion_criteria": ["Cite every reported repair."],
        },
    )
    store_interaction(claim_path, item)
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))

    app.run(timeout=10)

    assert not app.exception
    assert "### Proposed research plan" in [value.value for value in app.markdown]
    assert {button.label for button in app.button}.issuperset(
        {"Approve and run", "Cancel"}
    )
    assert next(button for button in app.button if button.label == "Rebuild index").disabled

    next(button for button in app.button if button.label == "Cancel").click().run(
        timeout=10
    )
    saved = load_research_history(claim_path, "CLM-SAMPLE-001")
    assert saved.interactions[0].status == "cancelled"


def test_citation_marker_contains_a_safe_source_tooltip():
    store = load_claim_store(SAMPLE_OUTPUT)
    source_ref = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"

    rendered = cited_answer_html(
        f"A bumper cover was invoiced. [{source_ref}]",
        [source_ref],
        store,
    )

    assert 'class="claim-citation"' in rendered
    assert ">[1]</sup>" in rendered
    assert source_ref in rendered
    assert "Repair Invoice" in rendered
    assert "front bumper cover" in rendered


def test_unresolved_citation_blocks_answer_rendering():
    store = load_claim_store(SAMPLE_OUTPUT)

    with pytest.raises(ChunkNotFoundError, match="Citation source not found"):
        cited_answer_html(
            f"This answer must not render. [{MISSING_SOURCE}]",
            [MISSING_SOURCE],
            store,
        )


def test_saved_answer_with_unresolved_citation_shows_error(monkeypatch, tmp_path):
    claim_path = copy_sample_claim(tmp_path)
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(claim_path.parent))
    store_interaction(
        claim_path,
        ResearchInteraction(
            claim_id="CLM-SAMPLE-001",
            status="completed",
            question="Render a missing source.",
            planning_enabled=False,
            output={
                "answer": f"Untrusted answer. [{MISSING_SOURCE}]",
                "source_refs": [MISSING_SOURCE],
                "evidence_sufficient": True,
            },
        ),
    )
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))

    app.run(timeout=10)

    assert not app.exception
    assert any(
        "Could not display saved research report: Citation source not found"
        in error.value
        for error in app.error
    )
    assert all("Untrusted answer" not in item.value for item in app.markdown)


def test_native_tool_trace_extracts_claim_search_details():
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    "claim_search",
                    {"query": "repair", "research_goal": "Find repairs"},
                    tool_call_id="search-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "claim_search",
                    [
                        {
                            "source_ref": (
                                "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
                            )
                        }
                    ],
                    tool_call_id="search-1",
                )
            ]
        ),
    ]

    trace = claim_search_trace(messages)

    assert trace == [
        {
            "query": "repair",
            "research_goal": "Find repairs",
            "source_refs": [
                "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
            ],
        }
    ]
    assert to_jsonable_python(messages)


@pytest.mark.parametrize("value", ["hi", "Hello!", "HEY", "Good morning"])
def test_simple_greetings_do_not_start_research(value):
    assert is_greeting(value)


def test_question_with_a_greeting_still_starts_research():
    assert not is_greeting("Hi, what caused the fire?")


def test_upload_mode_labels_remain_explicit():
    assert COMBINED_LABEL == "One combined claim PDF"
    assert SEPARATE_LABEL == "Separate document PDFs"
