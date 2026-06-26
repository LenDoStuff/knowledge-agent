# Claim KB

`claim_kb` turns one scanned insurance claim PDF into a structured, searchable
claim knowledge base.

This is the first module in the project. It is intentionally a proof of
concept: keep behavior explicit, keep the folder structure simple, and avoid
fallbacks or hidden recovery paths that make failures hard to see.

No frontend or UI is included.

## Input

The ingestion pipeline requires:

- `claim_id`: stable claim identifier, for example `CLM-001`
- `pdf_path`: path to one scanned claim PDF, for example
  `data/input/scanned_claim.pdf`

Run ingestion with:

```powershell
python -m claim_kb.ingest --claim-id CLM-001 --pdf-path data/input/scanned_claim.pdf
```

The PDF may contain many logical documents in one scanned file, such as FNOL,
emails, loss adjuster reports, invoices, or other claim documents.

## What It Does

1. Copies the original scanned PDF into the claim folder.
2. Runs Azure Document Intelligence OCR with `prebuilt-layout`.
3. Classifies each page using the current page, prior page, and current document
   context to decide whether it starts a new document or continues the previous
   document.
4. Writes split PDF files for each logical document.
5. Chunks OCR text by document and page range, assigning stable page IDs and
   chunk citation references.
6. Extracts metadata for each logical document:
   `id`, `title`, `summary`, `involved_parties`, `events`,
   `document_type`, and `page_range`.
   Event dates use nullable numeric `year`, `month`, and `day` fields, and every
   event cites a supporting chunk through `source_ref`.
7. Embeds each chunk with Snowflake Cortex `AI_EMBED`.
8. Stores chunk vectors in a claim-local Chroma vector store.
9. Exposes listing, search, and chunk-read functions through `claim_kb.api`.

If an expected service response, persisted file, metadata field, or vector-store
shape is missing, the module should fail clearly instead of silently inventing a
replacement.

## Output

By default, output is written under `data/claims/<claim_id>/`.

```text
data/claims/CLM-001/
  source/
    claim.pdf

  documents/
    DOC-001_fnol.pdf
    DOC-002_email.pdf
    DOC-003_loss_adjuster_report.pdf
    DOC-004_invoice.pdf

  manifest.json
  pages.jsonl
  chunks.jsonl
  run_log.json

  index/
    chroma/
```

Important output files:

- `source/claim.pdf`: preserved source PDF
- `documents/*.pdf`: logical document PDFs split from the source
- `manifest.json`: claim manifest and one metadata record per logical document
- `pages.jsonl`: OCR text and stable page IDs such as `CLM-001:p1`
- `chunks.jsonl`: chunk text, embeddings, page IDs, and stable citation
  references such as `CLM-001/DOC-001#DOC-001-CHUNK-001`
- `index/chroma/`: local vector index for retrieval
- `run_log.json`: step-level ingestion status

`manifest.json` records the documents, embedding provider, and model used for
the claim. Search uses that stored model when embedding the query, so query
vectors match the vectors already saved in Chroma.

## Examples

Synthetic output examples live in
[`../examples/claim_kb`](../examples/claim_kb/README.md). They are hand-written
documentation samples that show persisted file shapes without including real
claim data, generated PDFs, Chroma files, or service output.

## Programmatic API

Other modules should use the stable API facade:

```python
from claim_kb.api import ClaimKbApi

kb = ClaimKbApi()

claim_file = kb.ingest_claim_pdf(
    claim_id="CLM-001",
    pdf_path="data/input/scanned_claim.pdf",
)

documents = kb.list_claim_documents("CLM-001")

results = kb.search_claim_file(
    claim_id="CLM-001",
    query="repair invoice total",
    document_types=["invoice"],
    top_k=10,
)

chunk = kb.read_document_chunk(
    claim_id="CLM-001",
    document_id=results[0].document_id,
    chunk_id=results[0].chunk_id,
)
```

Convenience functions are also available:

```python
from claim_kb.api import (
    ingest_claim_pdf,
    list_claim_documents,
    read_document_chunk,
    search_claim_file,
)
```

For a local research agent that reads persisted output without Azure,
Snowflake, or Chroma, use the lexical knowledge store:

```python
from claim_kb import ClaimKbKnowledgeStore

store = ClaimKbKnowledgeStore("data/claims/CLM-001")
items = store.search("repair invoice total", top_k=8)
page = store.get_page(items[0].page_ids[0])
document = store.get_document(items[0].document_id)

print(items[0].source_ref)
```

`ClaimKbKnowledgeStore.search()` searches chunk text, document title, document
summary, and document type. It returns `KnowledgeItem` objects containing the
document metadata, page IDs, and citation-ready `source_ref`.

Core functions:

```python
ingest_claim_pdf(claim_id: str, pdf_path: str) -> StructuredClaimFile

list_claim_documents(claim_id: str) -> list[DocumentMetadata]

search_claim_file(
    claim_id: str,
    query: str,
    document_types: list[str] | None = None,
    top_k: int = 10,
) -> list[ChunkSearchResult]

read_document_chunk(
    claim_id: str,
    document_id: str,
    chunk_id: str,
) -> DocumentChunk
```

Search results include:

- `document_id`
- `chunk_id`
- `page_range`
- `text`
- `score`
- `document_type`

## Configuration

Azure OCR/classification authentication is keyless through Microsoft Entra ID
browser sign-in. The code uses `InteractiveBrowserCredential` and does not
require Azure API keys.

Embeddings are generated through Snowflake Cortex `AI_EMBED`. The code creates a
Snowpark session from your local Snowflake TOML connection config.

Required for live ingestion:

- `AZURE_AI_PROJECT_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`

Optional:

- `AZURE_TENANT_ID`: tenant to use for browser sign-in
- `SNOWFLAKE_CONNECTION_NAME`: local Snowflake connection name, defaults to
  `default`
- `SNOWFLAKE_EMBEDDING_MODEL`: Snowflake `AI_EMBED` model, defaults to
  `snowflake-arctic-embed-l-v2.0`
- `CLAIM_KB_DATA_ROOT`: output root, defaults to `data/claims`

The Document Intelligence endpoint must be a custom subdomain endpoint for
Microsoft Entra authentication, not a regional endpoint.

Example PowerShell configuration:

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT="https://example.services.ai.azure.com/api/projects/my-project"
$env:AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://my-doc-intel.cognitiveservices.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT="gpt-4.1"
$env:SNOWFLAKE_CONNECTION_NAME="default"
$env:SNOWFLAKE_EMBEDDING_MODEL="snowflake-arctic-embed-l-v2.0"
```

## Development

Install locally:

```powershell
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

The tests use mocked Azure and Snowflake clients plus a generated sample PDF, so
they do not open browser auth or call live Azure/Snowflake services.
