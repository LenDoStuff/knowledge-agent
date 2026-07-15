"""Tests for claim-ingestion CLI arguments and logging."""

import logging

import pytest

from knowledge_agent.claims.cli import build_parser, configure_logging


def test_cli_accepts_exactly_one_ingestion_input():
    parser = build_parser()

    pdf_args = parser.parse_args(
        ["--claim-id", "CLM-001", "--pdf-path", "claim.pdf"]
    )
    folder_args = parser.parse_args(
        [
            "--claim-id",
            "CLM-001",
            "--folder-path",
            "documents",
            "--log-level",
            "DEBUG",
        ]
    )

    assert pdf_args.pdf_path == "claim.pdf"
    assert pdf_args.knowledge_base == "custom"
    assert folder_args.folder_path == "documents"
    assert folder_args.log_level == "DEBUG"

    lightrag_args = parser.parse_args(
        [
            "--claim-id",
            "CLM-002",
            "--pdf-path",
            "claim.pdf",
            "--knowledge-base",
            "lightrag",
        ]
    )
    assert lightrag_args.knowledge_base == "lightrag"

    both_args = parser.parse_args(
        [
            "--claim-id",
            "CLM-003",
            "--pdf-path",
            "claim.pdf",
            "--knowledge-base",
            "both",
        ]
    )
    assert both_args.knowledge_base == "both"

    with pytest.raises(SystemExit):
        parser.parse_args(["--claim-id", "CLM-001"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--claim-id",
                "CLM-001",
                "--pdf-path",
                "claim.pdf",
                "--folder-path",
                "documents",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--claim-id",
                "CLM-001",
                "--pdf-path",
                "claim.pdf",
                "--embedding-mode",
                "none",
            ]
        )


def test_debug_logging_writes_to_selected_file(tmp_path):
    log_path = tmp_path / "claims.log"
    try:
        configure_logging("DEBUG", log_path)
        logging.getLogger("knowledge_agent.claims.test").debug("debug detail")
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert "debug detail" in log_path.read_text(encoding="utf-8")
        for logger_name in ("azure", "httpx", "openai"):
            assert logging.getLogger(logger_name).getEffectiveLevel() == logging.WARNING
    finally:
        logging.basicConfig(handlers=[logging.NullHandler()], force=True)
