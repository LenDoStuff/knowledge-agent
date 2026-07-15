"""Tests for research CLI arguments and logging."""

import logging

from knowledge_agent.research.cli import build_parser, configure_logging


def test_cli_uses_deep_agent_search_and_request_limits():
    args = build_parser().parse_args(
        [
            "--claim-path",
            "claim",
            "--question",
            "What happened?",
            "--max-searches",
            "3",
            "--request-limit",
            "8",
        ]
    )
    assert args.max_searches == 3
    assert args.request_limit == 8


def test_cli_logging_creates_fixed_file_and_appends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    try:
        configure_logging("INFO")
        logging.getLogger("knowledge_agent.research.test").info("first run")
        for handler in logging.getLogger().handlers:
            handler.flush()

        configure_logging("INFO")
        logging.getLogger("knowledge_agent.research.test").info("second run")
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_path = tmp_path / "logs" / "research.log"
        content = log_path.read_text(encoding="utf-8")
        assert "first run" in content
        assert "second run" in content
        for logger_name in ("azure", "httpx", "openai"):
            assert logging.getLogger(logger_name).getEffectiveLevel() == logging.WARNING
    finally:
        logging.basicConfig(handlers=[logging.NullHandler()], force=True)
