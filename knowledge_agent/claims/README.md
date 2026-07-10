# Claims

The claims package owns the complete persisted claim boundary: OCR, logical
document preparation, chunking, metadata extraction, indexing, and retrieval.
LLM-assisted document classification lives in
`knowledge_agent.agents.document_classifier` and is composed into ingestion
through explicit services.

## Inputs and output

Ingestion accepts either one combined PDF or a folder of already-separated PDFs.
Each claim is written to `CLAIM_DATA_ROOT/<claim_id>/`:

```text
source/claim.pdf       # combined-PDF input only
documents/*.pdf
manifest.json
pages.jsonl
chunks.jsonl
run_log.json
index/chroma/          # semantic claims only
```

`manifest.json` records `retrieval_mode` as `lexical` or `semantic`. Semantic
manifests also record the Snowflake embedding provider and model. Embedding
vectors live only in Chroma; they are not duplicated in `chunks.jsonl`.

## CLI

```powershell
python -m knowledge_agent.claims.cli `
  --claim-id CLM-001 `
  --pdf-path data/input/scanned_claim.pdf

python -m knowledge_agent.claims.cli `
  --claim-id CLM-002 `
  --folder-path examples/claims/sample_input `
  --log-level DEBUG
```

The `api_key` profile produces lexical claims. The `azure_project` profile uses
Snowflake Cortex `AI_EMBED` and writes a claim-local Chroma index.

Claims append logs to `logs/claims.log`. `INFO` records OCR counts, ingestion
steps, LLM request IDs, token usage, and latency. `DEBUG` additionally records
OCR-derived classifier prompts and parsed outputs, so debug logs may contain
claim data and must remain local.

For a direct two-document VS Code walkthrough that avoids pytest internals, see
[`DEBUGGING.md`](DEBUGGING.md).

## Programmatic use

The ingestion pipeline accepts explicit dependencies and does not close them.
The composition context owns live clients:

```python
from pathlib import Path

from knowledge_agent.claims.config import load_claim_settings
from knowledge_agent.claims.dependencies import live_ingestion_services
from knowledge_agent.claims.pipeline import ingest_claim_pdf
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import load_llm_settings

profile = load_profile()
claim_settings = load_claim_settings()
llm_settings = load_llm_settings(profile)

with live_ingestion_services("CLM-001", claim_settings, llm_settings) as services:
    manifest = ingest_claim_pdf(
        "CLM-001",
        Path("claim.pdf"),
        claim_settings.data_root,
        services,
    )
```

Use the same store interface for both retrieval modes:

```python
from knowledge_agent.claims.config import load_claim_settings
from knowledge_agent.claims.dependencies import open_claim_store
from knowledge_agent.claims.store import get_document, get_page, search_claim

with open_claim_store("data/claims/CLM-001", load_claim_settings()) as store:
    results = search_claim(store, "repair invoice total", top_k=8)
    document = get_document(store, results[0].document_id)
    page = get_page(store, results[0].page_ids[0])
```

Every result includes document metadata, page IDs, chunk text, score, and the
stable citation-ready `source_ref`.
