import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from knowledge_agent.app import (
    COMBINED_LABEL,
    SEPARATE_LABEL,
    _cited_answer_html,
    _conversation_history,
    _document_rows,
    _is_greeting,
    _party_rows,
    _timeline_rows,
    discover_claims,
    ingest_uploads,
    validate_uploads,
)
from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.errors import ChunkNotFoundError
from knowledge_agent.claims.store import load_claim_store
from knowledge_agent.llm.config import LlmSettings


REPO_ROOT = Path(__file__).parents[1]
SAMPLE_OUTPUT = REPO_ROOT / "examples" / "claims" / "sample_output"
MISSING_SOURCE = "CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"


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


def saved_research_answer() -> dict[str, object]:
    return {
        "question": "What was repaired?",
        "answer": "The claim knowledge base does not contain enough evidence.",
        "plan": {
            "objectives": ["Identify repaired items."],
            "queries": [
                {
                    "query": "repair invoice",
                    "research_goal": "Find repaired items.",
                }
            ],
        },
        "searches": [],
        "gap_reviews": [],
        "steps": [],
        "findings": [],
        "source_refs": [],
    }


def saved_audit_trail() -> dict[str, object]:
    source_ref = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"
    return {
        "entries": [
            {
                "kind": "llm",
                "operation": "plan_research",
                "response_model": "ResearchPlan",
                "system_prompt": "Exact system audit prompt.",
                "user_prompt": "Exact user audit prompt.",
                "result": {
                    "objectives": ["Identify repaired items."],
                    "queries": [
                        {
                            "query": "repair invoice",
                            "research_goal": "Find repaired items.",
                        }
                    ],
                },
            },
            {
                "kind": "tool",
                "tool_name": "claim_search",
                "query": {
                    "query": "repair invoice",
                    "research_goal": "Find repaired items.",
                },
                "top_k": 8,
                "result": [
                    {
                        "document_id": "DOC-002",
                        "document_type": "invoice",
                        "document_title": "Repair Invoice",
                        "page_ids": ["CLM-SAMPLE-001:p2"],
                        "source_ref": source_ref,
                        "text": "Parts: front bumper cover",
                    }
                ],
            },
        ]
    }


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
        "knowledge_agent.app.live_ingestion_services",
        fake_services,
    )
    monkeypatch.setattr(
        "knowledge_agent.app.ingest_claim_pdf",
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
        ("Audit mode", False)
    ]


def test_streamlit_app_renders_saved_audit_trail(monkeypatch):
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(SAMPLE_OUTPUT.parent))
    research = saved_research_answer()
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.session_state["research_audit_mode"] = True
    app.session_state["claim_chat_histories"] = {
        "CLM-SAMPLE-001": [
            {
                "role": "assistant",
                "content": research["answer"],
                "research": research,
                "audit": saved_audit_trail(),
            }
        ]
    }

    app.run(timeout=10)

    assert not app.exception
    assert any(
        "Audit traces can contain full claim evidence" in item.value
        for item in app.warning
    )
    assert any(item.label == "Audit trail (2 entries)" for item in app.expander)
    assert "Exact system audit prompt." in [item.value for item in app.code]
    assert "Exact user audit prompt." in [item.value for item in app.code]

    clear_chat = next(button for button in app.button if button.label == "Clear chat")
    clear_chat.click().run(timeout=10)

    assert app.session_state["claim_chat_histories"]["CLM-SAMPLE-001"] == []


def test_streamlit_app_identifies_turn_without_captured_audit(monkeypatch):
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(SAMPLE_OUTPUT.parent))
    research = saved_research_answer()
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.session_state["research_audit_mode"] = True
    app.session_state["claim_chat_histories"] = {
        "CLM-SAMPLE-001": [
            {
                "role": "assistant",
                "content": research["answer"],
                "research": research,
            }
        ]
    }

    app.run(timeout=10)

    assert not app.exception
    assert "Audit mode was not enabled for this turn." in [
        item.value for item in app.caption
    ]


def test_saved_failed_turn_renders_audit_and_stays_out_of_model_history(monkeypatch):
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(SAMPLE_OUTPUT.parent))
    messages = [
        {
            "role": "user",
            "content": "What failed?",
            "exclude_from_history": True,
        },
        {
            "role": "assistant",
            "content": "Research failed: model unavailable",
            "research_error": "model unavailable",
            "exclude_from_history": True,
            "audit": {
                "entries": [
                    {
                        "kind": "llm",
                        "operation": "plan_research",
                        "response_model": "ResearchPlan",
                        "system_prompt": "Failed system prompt.",
                        "user_prompt": "Failed user prompt.",
                        "error": "model unavailable",
                    }
                ]
            },
        },
    ]
    assert _conversation_history(messages) == []

    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.session_state["research_audit_mode"] = True
    app.session_state["claim_chat_histories"] = {"CLM-SAMPLE-001": messages}

    app.run(timeout=10)

    assert not app.exception
    assert "Research failed: model unavailable" in [item.value for item in app.error]
    assert any(item.label == "Audit trail (1 entry)" for item in app.expander)
    assert "Failed system prompt." in [item.value for item in app.code]


def test_claim_overview_aggregates_timeline_and_parties():
    store = load_claim_store(SAMPLE_OUTPUT)

    timeline = _timeline_rows(store)
    parties = _party_rows(store)

    assert timeline[0]["Date"] == "2026-06-01"
    assert timeline[-1]["Date"] == "Undated"
    assert {party["Party"] for party in parties} == {
        "Casey Sample",
        "Example Mutual",
        "Sample Body Shop",
    }


def test_document_inventory_summarizes_extracted_metadata():
    rows = _document_rows(load_claim_store(SAMPLE_OUTPUT))

    assert rows == [
        {
            "ID": "DOC-001",
            "Type": "fnol",
            "Title": "First Notice of Loss",
            "Pages": "1",
            "Parties": 2,
            "Events": 2,
            "Evidence chunks": 1,
            "File": "DOC-001_fnol.pdf",
        },
        {
            "ID": "DOC-002",
            "Type": "invoice",
            "Title": "Repair Invoice",
            "Pages": "2",
            "Parties": 2,
            "Events": 1,
            "Evidence chunks": 1,
            "File": "DOC-002_invoice.pdf",
        },
    ]


def test_citation_marker_contains_a_safe_source_tooltip():
    store = load_claim_store(SAMPLE_OUTPUT)
    source_ref = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"

    rendered = _cited_answer_html(
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
        _cited_answer_html(
            f"This answer must not render. [{MISSING_SOURCE}]",
            [MISSING_SOURCE],
            store,
        )


def test_saved_answer_with_unresolved_citation_shows_error(monkeypatch):
    monkeypatch.setenv("CLAIM_DATA_ROOT", str(SAMPLE_OUTPUT.parent))
    research = {
        "question": "What was repaired?",
        "answer": f"Untrusted answer. [{MISSING_SOURCE}]",
        "plan": {
            "objectives": ["Answer the question."],
            "queries": [
                {
                    "query": "repair",
                    "research_goal": "Find repair evidence.",
                }
            ],
        },
        "searches": [],
        "gap_reviews": [],
        "steps": [],
        "findings": [],
        "source_refs": [MISSING_SOURCE],
    }
    app = AppTest.from_file(str(REPO_ROOT / "knowledge_agent" / "app.py"))
    app.session_state["claim_chat_histories"] = {
        "CLM-SAMPLE-001": [
            {
                "role": "assistant",
                "content": research["answer"],
                "research": research,
            }
        ]
    }

    app.run(timeout=10)

    assert not app.exception
    assert any(
        "Could not display saved research answer: Citation source not found"
        in error.value
        for error in app.error
    )
    assert all("Untrusted answer" not in item.value for item in app.markdown)


@pytest.mark.parametrize("value", ["hi", "Hello!", "HEY", "Good morning"])
def test_simple_greetings_do_not_start_research(value):
    assert _is_greeting(value)


def test_question_with_a_greeting_still_starts_research():
    assert not _is_greeting("Hi, what caused the fire?")


def test_upload_mode_labels_remain_explicit():
    assert COMBINED_LABEL == "One combined claim PDF"
    assert SEPARATE_LABEL == "Separate document PDFs"
