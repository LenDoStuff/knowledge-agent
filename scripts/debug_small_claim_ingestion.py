"""Run the small paid API-key ingestion directly under a debugger."""

from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from knowledge_agent.claims.cli import configure_logging
from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import live_ingestion_services
from knowledge_agent.claims.filesystem import read_jsonl
from knowledge_agent.claims.models import ClaimManifest
from knowledge_agent.claims.pipeline import ingest_claim_folder
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import LlmSettings


REPO_ROOT = Path(__file__).parents[1]
SAMPLE_INPUT = REPO_ROOT / "examples" / "claims" / "sample_input"
LIVE_DATA_ROOT = REPO_ROOT / "data" / "live-runs"
CLAIM_ID = "API-KEY-SMALL"
PAID_RUN_ENV = "RUN_API_KEY_SMALL_INGESTION"
SMALL_DOCUMENTS = (
    "00_claim_file_index.pdf",
    "01_fnol_and_broker_notice.pdf",
)
EXPECTED_DOCUMENTS = 2
EXPECTED_PAGES = 4


def main() -> None:
    _require_paid_run()
    profile = load_profile()
    if profile != "api_key":
        raise SystemExit(
            "Small ingestion debugging requires "
            "KNOWLEDGE_AGENT_PROFILE=api_key"
        )

    output_path = LIVE_DATA_ROOT / CLAIM_ID
    if output_path.exists():
        shutil.rmtree(output_path)
    debug_log_path = output_path / "debug.log"

    claim_settings = replace(ClaimSettings.from_env(), data_root=LIVE_DATA_ROOT)
    llm_settings = LlmSettings.from_env(profile)
    configure_logging("DEBUG", debug_log_path)

    with TemporaryDirectory(prefix="knowledge-agent-small-") as temporary_dir:
        input_path = Path(temporary_dir)
        _copy_small_input(input_path)

        with live_ingestion_services(
            CLAIM_ID,
            claim_settings,
            llm_settings,
        ) as services:
            manifest = ingest_claim_folder(
                claim_id=CLAIM_ID,
                folder_path=input_path,
                data_root=LIVE_DATA_ROOT,
                services=services,
            )

    _validate_output(manifest, output_path)
    print(f"Small ingestion complete: {output_path}")
    print(f"Debug log: {debug_log_path}")


def _require_paid_run() -> None:
    if os.getenv(PAID_RUN_ENV) != "1":
        raise SystemExit(
            f"Paid API calls are disabled. Set {PAID_RUN_ENV}=1 explicitly."
        )


def _copy_small_input(input_path: Path) -> None:
    for file_name in SMALL_DOCUMENTS:
        source_path = SAMPLE_INPUT / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Sample document does not exist: {source_path}")
        shutil.copy2(source_path, input_path / file_name)


def _validate_output(manifest: ClaimManifest, output_path: Path) -> None:
    pages = read_jsonl(output_path / "pages.jsonl")
    problems: list[str] = []
    if manifest.retrieval_mode != "lexical":
        problems.append(f"retrieval mode is {manifest.retrieval_mode!r}, not 'lexical'")
    if len(manifest.documents) != EXPECTED_DOCUMENTS:
        problems.append(
            f"produced {len(manifest.documents)} documents, not {EXPECTED_DOCUMENTS}"
        )
    if len(pages) != EXPECTED_PAGES:
        problems.append(f"produced {len(pages)} pages, not {EXPECTED_PAGES}")
    if (output_path / "index").exists():
        problems.append("created an index directory for a lexical claim")
    if problems:
        raise RuntimeError("Small ingestion validation failed: " + "; ".join(problems))


if __name__ == "__main__":
    main()
