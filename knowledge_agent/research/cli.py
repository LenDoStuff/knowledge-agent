"""Command-line entry point for claim research."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from knowledge_agent.claims.config import load_claim_settings
from knowledge_agent.claims.dependencies import open_claim_store
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import load_llm_settings
from knowledge_agent.llm.providers import open_agent_runtime
from knowledge_agent.agents.claim_researcher import run_claim_research


LOG_PATH = Path("logs") / "research.log"
QUIET_LOGGERS = ("azure", "httpcore", "httpx", "msal", "openai", "snowflake")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research one question against a persisted claim."
    )
    parser.add_argument("--claim-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-searches", type=int, default=6)
    parser.add_argument("--request-limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--log-level",
        choices=["INFO", "DEBUG"],
        default="INFO",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    profile = load_profile()
    llm_settings = load_llm_settings(profile)
    claim_settings = load_claim_settings()
    with (
        open_agent_runtime(llm_settings) as runtime,
        open_claim_store(
            args.claim_path,
            claim_settings,
            runtime=runtime,
            llm_settings=llm_settings,
        ) as store,
    ):
        result = run_claim_research(
            runtime=runtime,
            store=store,
            question=args.question,
            max_searches=args.max_searches,
            request_limit=args.request_limit,
            top_k=args.top_k,
        )

    print(result.output.answer)
    print("\nSources:")
    for source_ref in result.output.source_refs:
        print(f"- {source_ref}")
    usage = result.usage
    print(
        f"\nUsage: {usage.requests} model requests, "
        f"{usage.tool_calls} tool calls, {usage.total_tokens} tokens"
    )


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
