"""Command-line entry point for claim ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from knowledge_agent.claims.config import load_claim_settings
from knowledge_agent.claims.dependencies import live_ingestion_services
from knowledge_agent.claims.pipeline import ingest_claim_folder, ingest_claim_pdf
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import load_llm_settings


LOG_PATH = Path("logs") / "claims.log"
QUIET_LOGGERS = ("azure", "httpcore", "httpx", "msal", "openai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a combined claim PDF or a folder of document PDFs."
    )
    parser.add_argument("--claim-id", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf-path")
    input_group.add_argument("--folder-path")
    parser.add_argument(
        "--knowledge-base",
        choices=["custom", "lightrag", "both"],
        default="custom",
    )
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
    claim_settings = load_claim_settings()
    llm_settings = load_llm_settings(profile)
    with live_ingestion_services(
        args.claim_id,
        claim_settings,
        llm_settings,
        args.knowledge_base,
    ) as services:
        if args.pdf_path is not None:
            manifest = ingest_claim_pdf(
                claim_id=args.claim_id,
                pdf_path=Path(args.pdf_path),
                data_root=claim_settings.data_root,
                services=services,
            )
        else:
            manifest = ingest_claim_folder(
                claim_id=args.claim_id,
                folder_path=Path(args.folder_path),
                data_root=claim_settings.data_root,
                services=services,
            )
    print(manifest.model_dump_json(indent=2))


def configure_logging(level: str, log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
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
