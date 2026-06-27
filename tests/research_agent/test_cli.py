import logging

from research_agent.cli import configure_logging


def test_cli_logging_creates_fixed_file_and_appends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    try:
        configure_logging("INFO")
        logging.getLogger("research_agent.test").info("first run")
        for handler in logging.getLogger().handlers:
            handler.flush()

        configure_logging("INFO")
        logging.getLogger("research_agent.test").info("second run")
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_path = tmp_path / "logs" / "research_agent.log"
        content = log_path.read_text(encoding="utf-8")
        assert "first run" in content
        assert "second run" in content
        for logger_name in ("azure", "httpx", "openai"):
            assert logging.getLogger(logger_name).getEffectiveLevel() == logging.WARNING
    finally:
        logging.basicConfig(handlers=[logging.NullHandler()], force=True)
