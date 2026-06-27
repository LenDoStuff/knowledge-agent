"""Command-line entry point for local claim research."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research_agent.agent import run_claim_research
from research_agent.bootstrap import live_research_llm


LOG_PATH = Path("logs") / "research_agent.log"
QUIET_LOGGERS = ("azure", "httpcore", "httpx", "msal", "openai")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research one question against a persisted claim KB."
    )
    parser.add_argument("--claim-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--breadth", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--log-level",
        choices=["INFO", "DEBUG"],
        default="INFO",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    with live_research_llm() as llm:
        answer = run_claim_research(
            claim_path=args.claim_path,
            question=args.question,
            llm=llm,
            breadth=args.breadth,
            depth=args.depth,
            top_k=args.top_k,
        )

    print(answer.answer)
    print("\nSources:")
    for source_ref in answer.source_refs:
        print(f"- {source_ref}")


def configure_logging(level: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        encoding="utf-8",
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
