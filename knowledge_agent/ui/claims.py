"""Claim discovery, upload validation, and ingestion for the Streamlit UI."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, Sequence

from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import live_ingestion_services
from knowledge_agent.claims.filesystem import claim_root, read_json, safe_claim_id
from knowledge_agent.claims.models import ClaimManifest, KnowledgeBaseEngine
from knowledge_agent.claims.pipeline import ingest_claim_folder, ingest_claim_pdf
from knowledge_agent.llm.config import LlmSettings


UploadMode = Literal["combined", "separate"]
COMBINED_LABEL = "One combined claim PDF"
SEPARATE_LABEL = "Separate document PDFs"


class UploadedPdf(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


@dataclass(frozen=True)
class ClaimEntry:
    path: Path
    manifest: ClaimManifest


@dataclass(frozen=True)
class InvalidClaim:
    path: Path
    error: str


def discover_claims(data_root: Path) -> tuple[list[ClaimEntry], list[InvalidClaim]]:
    if not data_root.exists():
        return [], []
    claims: list[ClaimEntry] = []
    invalid: list[InvalidClaim] = []
    for path in sorted(
        (item for item in data_root.iterdir() if item.is_dir()),
        key=lambda item: item.name.casefold(),
    ):
        try:
            manifest_data = read_json(path / "manifest.json")
            if not isinstance(manifest_data, dict):
                raise ValueError("manifest.json must contain a JSON object")
            manifest = ClaimManifest.model_validate(manifest_data)
        except Exception as exc:
            invalid.append(InvalidClaim(path=path, error=str(exc)))
        else:
            claims.append(ClaimEntry(path=path, manifest=manifest))
    return claims, invalid


def validate_uploads(mode: UploadMode, uploads: Sequence[UploadedPdf]) -> None:
    if not uploads:
        raise ValueError("Select at least one PDF")
    if mode == "combined" and len(uploads) != 1:
        raise ValueError("Combined claim mode requires exactly one PDF")
    names: set[str] = set()
    for upload in uploads:
        name = Path(upload.name).name
        if not name or Path(name).suffix.casefold() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {upload.name}")
        key = name.casefold()
        if key in names:
            raise ValueError(f"Duplicate uploaded file name: {name}")
        names.add(key)


def ingest_uploads(
    claim_id: str,
    mode: UploadMode,
    uploads: Sequence[UploadedPdf],
    claim_settings: ClaimSettings,
    llm_settings: LlmSettings,
    knowledge_base: KnowledgeBaseEngine = "custom",
) -> ClaimManifest:
    claim_id = safe_claim_id(claim_id)
    validate_uploads(mode, uploads)
    output_path = claim_root(claim_settings.data_root, claim_id)
    if output_path.exists():
        raise FileExistsError(
            f"Claim {claim_id!r} already exists at {output_path}. "
            "Choose a different claim ID."
        )

    try:
        with TemporaryDirectory(prefix="knowledge-agent-upload-") as temporary_dir:
            upload_root = Path(temporary_dir)
            paths = []
            for upload in uploads:
                path = upload_root / Path(upload.name).name
                path.write_bytes(upload.getvalue())
                paths.append(path)
            with live_ingestion_services(
                claim_id,
                claim_settings,
                llm_settings,
                knowledge_base,
            ) as services:
                if mode == "combined":
                    return ingest_claim_pdf(
                        claim_id,
                        paths[0],
                        claim_settings.data_root,
                        services,
                    )
                return ingest_claim_folder(
                    claim_id,
                    upload_root,
                    claim_settings.data_root,
                    services,
                )
    except Exception:
        if output_path.exists():
            shutil.rmtree(output_path)
        raise
