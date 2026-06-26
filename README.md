# Knowledge Agent

Proof-of-concept backend modules for turning claim material into structured,
retrievable knowledge.

This repo should stay simple while the shape of the product is still being
discovered. Keep module boundaries clear, make behavior explicit, and avoid
hidden fallbacks or speculative abstractions. See [AGENTS.md](AGENTS.md) for the
working code mantra.

## Modules

| Module | Status | Purpose |
| --- | --- | --- |
| [`claim_kb`](claim_kb/README.md) | Current | Ingests scanned insurance claim PDFs into a structured, searchable claim knowledge base. |
| `research_agent` | Planned | Future module for claim research workflows. Not implemented yet. |

## Folder Structure

```text
.
  AGENTS.md
  README.md
  pyproject.toml
  claim_kb/
    README.md
    *.py
  tests/
    claim_kb/
      test_*.py
      conftest.py
```

Generated and local-only folders such as `.venv/`, `.pytest_cache/`,
`claim_kb.egg-info/`, `__pycache__/`, and `data/claims/` are ignored.

## Current Module

The only implemented module is [`claim_kb`](claim_kb/README.md). It preserves a
source claim PDF, runs OCR, splits logical documents, extracts metadata, chunks
text, embeds chunks with Snowflake Cortex, stores vectors in Chroma, and exposes
retrieval APIs for later modules.

Run ingestion with:

```powershell
python -m claim_kb.ingest --claim-id CLM-001 --pdf-path data/input/scanned_claim.pdf
```

Run tests with:

```powershell
python -m pytest
```

The future `research_agent` module should be added only when it is ready to be
implemented. Do not add placeholder code or package scaffolding for it yet.
