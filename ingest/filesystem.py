"""Claim KB filesystem layout and JSON persistence."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from ingest.exceptions import ClaimNotFoundError
from ingest.schemas import DocumentChunk, StructuredClaimFile


CLAIM_SUBDIRS = [
    "documents",
]


def safe_claim_id(claim_id: str) -> str:
    claim_id = claim_id.strip()
    if not claim_id:
        raise ValueError("claim_id cannot be empty")
    if any(sep in claim_id for sep in ("/", "\\")) or claim_id in {".", ".."}:
        raise ValueError("claim_id cannot contain path separators")
    return claim_id


def claim_root(data_root: Path, claim_id: str) -> Path:
    return data_root / safe_claim_id(claim_id)


def ensure_claim_dirs(data_root: Path, claim_id: str) -> Path:
    root = claim_root(data_root, claim_id)
    for subdir in CLAIM_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def require_claim_root(data_root: Path, claim_id: str) -> Path:
    root = claim_root(data_root, claim_id)
    if not root.exists():
        raise ClaimNotFoundError(f"Claim not found: {claim_id}")
    return root


def preserve_original_pdf(pdf_path: Path, root: Path) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    destination = root / "source" / "claim.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, destination)
    return destination


def preserve_document_pdf(pdf_path: Path, root: Path) -> Path:
    destination = root / "documents" / pdf_path.name
    shutil.copy2(pdf_path, destination)
    return destination


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, items: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_claim_metadata(root: Path, claim_file: StructuredClaimFile) -> None:
    write_json(root / "manifest.json", claim_file.model_dump(mode="json"))


def read_claim_metadata(data_root: Path, claim_id: str) -> StructuredClaimFile:
    root = require_claim_root(data_root, claim_id)
    path = root / "manifest.json"
    if not path.exists():
        raise ClaimNotFoundError(f"Claim metadata not found for claim: {claim_id}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid claim metadata: {path}")
    return StructuredClaimFile.model_validate(data)


def read_chunks(data_root: Path, claim_id: str) -> list[DocumentChunk]:
    root = require_claim_root(data_root, claim_id)
    return [
        DocumentChunk.model_validate(row)
        for row in read_jsonl(root / "chunks.jsonl")
    ]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Cannot create slug from empty value")
    return slug
