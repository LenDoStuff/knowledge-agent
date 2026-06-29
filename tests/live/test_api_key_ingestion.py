from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from knowledge_agent.claims.cli import configure_logging
from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import live_ingestion_services
from knowledge_agent.claims.filesystem import read_jsonl
from knowledge_agent.claims.pipeline import ingest_claim_folder
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import LlmSettings


pytestmark = pytest.mark.live_api_key_ingestion

REPO_ROOT = Path(__file__).parents[2]
SAMPLE_INPUT = REPO_ROOT / "examples" / "claims" / "sample_input"
LIVE_DATA_ROOT = REPO_ROOT / "data" / "live-runs"
SMALL_DOCUMENTS = (
    "00_claim_file_index.pdf",
    "01_fnol_and_broker_notice.pdf",
)


@pytest.mark.skipif(
    os.getenv("RUN_API_KEY_SMALL_INGESTION") != "1",
    reason="set RUN_API_KEY_SMALL_INGESTION=1 to run the paid small ingestion",
)
def test_api_key_small_ingestion_live(tmp_path, monkeypatch):
    input_path = tmp_path / "small-input"
    input_path.mkdir()
    for file_name in SMALL_DOCUMENTS:
        shutil.copy2(SAMPLE_INPUT / file_name, input_path / file_name)

    _run_live_ingestion(
        claim_id="API-KEY-SMALL",
        input_path=input_path,
        expected_documents=2,
        expected_pages=4,
        monkeypatch=monkeypatch,
    )


@pytest.mark.skipif(
    os.getenv("RUN_API_KEY_FULL_INGESTION") != "1",
    reason="set RUN_API_KEY_FULL_INGESTION=1 to run the paid full ingestion",
)
def test_api_key_full_ingestion_live(monkeypatch):
    _run_live_ingestion(
        claim_id="API-KEY-FULL",
        input_path=SAMPLE_INPUT,
        expected_documents=14,
        expected_pages=27,
        monkeypatch=monkeypatch,
    )


def _run_live_ingestion(
    *,
    claim_id: str,
    input_path: Path,
    expected_documents: int,
    expected_pages: int,
    monkeypatch,
) -> None:
    profile = load_profile()
    if profile != "api_key":
        pytest.fail(
            "Live API-key ingestion requires KNOWLEDGE_AGENT_PROFILE=api_key"
        )

    output_path = LIVE_DATA_ROOT / claim_id
    if output_path.exists():
        shutil.rmtree(output_path)
    debug_log_path = output_path / "debug.log"

    def reject_semantic_dependency(*args, **kwargs):
        pytest.fail("api_key ingestion must not construct Snowflake or Chroma")

    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.SnowflakeAiEmbedder",
        reject_semantic_dependency,
    )
    monkeypatch.setattr(
        "knowledge_agent.claims.dependencies.ChromaVectorStore",
        reject_semantic_dependency,
    )

    claim_settings = replace(ClaimSettings.from_env(), data_root=LIVE_DATA_ROOT)
    llm_settings = LlmSettings.from_env(profile)
    configure_logging("DEBUG", debug_log_path)
    try:
        with live_ingestion_services(
            claim_id,
            claim_settings,
            llm_settings,
        ) as services:
            manifest = ingest_claim_folder(
                claim_id=claim_id,
                folder_path=input_path,
                data_root=LIVE_DATA_ROOT,
                services=services,
            )
        _flush_logs()
        _assert_live_output(
            output_path=output_path,
            debug_log_path=debug_log_path,
            manifest=manifest,
            expected_documents=expected_documents,
            expected_pages=expected_pages,
        )
        print(f"Live ingestion output: {output_path}")
        print(f"Debug log: {debug_log_path}")
    finally:
        _flush_logs()
        logging.basicConfig(handlers=[logging.NullHandler()], force=True)


def _assert_live_output(
    *,
    output_path: Path,
    debug_log_path: Path,
    manifest,
    expected_documents: int,
    expected_pages: int,
) -> None:
    assert manifest.retrieval_mode == "lexical"
    assert len(manifest.documents) == expected_documents
    assert len(manifest.source_files) == expected_documents
    assert manifest.embedding_provider is None
    assert manifest.embedding_model is None
    assert not (output_path / "index").exists()

    pages = read_jsonl(output_path / "pages.jsonl")
    chunks = read_jsonl(output_path / "chunks.jsonl")
    assert len(pages) == expected_pages
    assert len(chunks) == manifest.chunk_count
    assert chunks

    page_ids = {page["page_id"] for page in pages}
    for document in manifest.documents:
        assert document.title.strip()
        assert document.document_type.strip()
        assert document.file_name.strip()
    for chunk in chunks:
        assert chunk["source_ref"] == (
            f"{manifest.claim_id}/{chunk['document_id']}#{chunk['chunk_id']}"
        )
        assert set(chunk["page_ids"]) <= page_ids
        assert chunk["text"].strip()

    run_log = json.loads((output_path / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["entries"]
    assert all(entry["status"] == "succeeded" for entry in run_log["entries"])

    debug_log = debug_log_path.read_text(encoding="utf-8")
    for expected_message in (
        "ocr_complete",
        "llm_request provider=nvidia",
        "llm_response provider=nvidia",
        "claim_classifier_prompt",
        "claim_classifier_output",
        "ingestion_step_complete",
    ):
        assert expected_message in debug_log


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
