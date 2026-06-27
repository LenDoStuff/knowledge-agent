"""Command-line entry point for Claim KB ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from ingest.bootstrap import build_live_ingestion_services
from ingest.config import ClaimKbSettings
from ingest.ingest import (
    ingest_claim_folder_with_services,
    ingest_claim_pdf_with_services,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a combined claim PDF or a folder of document PDFs."
    )
    parser.add_argument("--claim-id", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf-path")
    input_group.add_argument("--folder-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    settings = ClaimKbSettings.from_env()
    services = build_live_ingestion_services(args.claim_id, settings)
    if args.pdf_path is not None:
        claim_file = ingest_claim_pdf_with_services(
            claim_id=args.claim_id,
            pdf_path=Path(args.pdf_path),
            data_root=settings.data_root,
            services=services,
        )
    else:
        claim_file = ingest_claim_folder_with_services(
            claim_id=args.claim_id,
            folder_path=Path(args.folder_path),
            data_root=settings.data_root,
            services=services,
        )
    print(claim_file.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
