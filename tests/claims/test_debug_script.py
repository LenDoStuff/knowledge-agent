"""Tests for the documented claim-debugging walkthrough."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_agent.claims.config import ClaimSettings
from scripts import debug_small_claim_ingestion


def test_main_rejects_disabled_paid_run(monkeypatch):
    monkeypatch.delenv(debug_small_claim_ingestion.PAID_RUN_ENV, raising=False)
    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "load_profile",
        lambda: pytest.fail("profile must not load before paid-run confirmation"),
    )

    with pytest.raises(SystemExit, match="Paid API calls are disabled"):
        debug_small_claim_ingestion.main()


def test_main_requires_api_key_profile(monkeypatch):
    monkeypatch.setenv(debug_small_claim_ingestion.PAID_RUN_ENV, "1")
    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "load_profile",
        lambda: "azure_project",
    )

    with pytest.raises(SystemExit, match="KNOWLEDGE_AGENT_PROFILE=api_key"):
        debug_small_claim_ingestion.main()


def test_main_runs_direct_small_ingestion(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(debug_small_claim_ingestion.PAID_RUN_ENV, "1")
    monkeypatch.setattr(debug_small_claim_ingestion, "load_profile", lambda: "api_key")
    monkeypatch.setattr(debug_small_claim_ingestion, "LIVE_DATA_ROOT", tmp_path)
    monkeypatch.setattr(debug_small_claim_ingestion, "_copy_small_input", lambda _: None)

    settings = ClaimSettings(
        data_root=Path("unused"),
        document_intelligence_endpoint="https://example.test",
        document_intelligence_api_key="secret",
        document_intelligence_connection_name=None,
        snowflake_connection_name="unused",
        snowflake_embedding_model="unused",
    )
    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "ClaimSettings",
        SimpleNamespace(from_env=lambda: settings),
    )
    llm_settings = SimpleNamespace(profile="api_key")
    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "LlmSettings",
        SimpleNamespace(from_env=lambda profile: llm_settings),
    )

    configured_logs = []
    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "configure_logging",
        lambda level, path: configured_logs.append((level, path)),
    )

    services = object()

    @contextmanager
    def fake_live_services(claim_id, claim_settings, received_llm_settings):
        assert claim_id == debug_small_claim_ingestion.CLAIM_ID
        assert claim_settings.data_root == tmp_path
        assert received_llm_settings is llm_settings
        yield services

    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "live_ingestion_services",
        fake_live_services,
    )

    output_path = tmp_path / debug_small_claim_ingestion.CLAIM_ID
    output_path.mkdir()
    stale_file = output_path / "stale.txt"
    stale_file.write_text("old output", encoding="utf-8")

    def fake_ingest_claim_folder(**kwargs):
        assert kwargs["claim_id"] == debug_small_claim_ingestion.CLAIM_ID
        assert kwargs["data_root"] == tmp_path
        assert kwargs["services"] is services
        assert kwargs["folder_path"].is_dir()
        assert not stale_file.exists()
        output_path.mkdir(parents=True, exist_ok=True)
        pages = "".join(f'{{"page_id":"p{number}"}}\n' for number in range(1, 5))
        (output_path / "pages.jsonl").write_text(pages, encoding="utf-8")
        return SimpleNamespace(
            retrieval_mode="lexical",
            documents=[object(), object()],
        )

    monkeypatch.setattr(
        debug_small_claim_ingestion,
        "ingest_claim_folder",
        fake_ingest_claim_folder,
    )

    debug_small_claim_ingestion.main()

    assert configured_logs == [("DEBUG", output_path / "debug.log")]
    assert not (output_path / "index").exists()
    assert "Small ingestion complete" in capsys.readouterr().out
