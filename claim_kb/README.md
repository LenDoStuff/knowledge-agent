# Claim KB

`claim_kb` turns scanned insurance claim documents into a structured,
searchable claim knowledge base. It accepts either one combined PDF or a folder
of PDFs that are already separate documents.

This is the first module in the project. It is intentionally a proof of
concept: keep behavior explicit, keep the folder structure simple, and avoid
fallbacks or hidden recovery paths that make failures hard to see.

No frontend or UI is included.

## Input

Every ingestion requires a stable `claim_id`, for example `CLM-001`, and exactly
one input:

- `pdf_path`: one combined claim PDF that must be split into logical documents
- `folder_path`: a folder whose top-level PDFs are already separate documents

Run combined-PDF ingestion with:

```powershell
python -m claim_kb.cli --claim-id CLM-001 --pdf-path data/input/scanned_claim.pdf
```

`KNOWLEDGE_AGENT_MODE` determines the complete runtime profile. `home` uses
OpenRouter, API-key Document Intelligence, and keyword retrieval without an
index. `work` uses Azure AI Projects with browser authentication, resolves
Document Intelligence from a named project connection, and builds Snowflake
embeddings plus a Chroma index.

The PDF may contain many logical documents in one scanned file, such as FNOL,
emails, loss adjuster reports, invoices, or other claim documents.

Run folder ingestion with:

```powershell
python -m claim_kb.cli `
  --claim-id PROP-B2B-2026-0417 `
  --folder-path examples/claim_kb/sample_input
```

Folder ingestion processes only top-level `.pdf` files. Each PDF is OCR'd and
classified as one complete document; it is never split. Documents are sorted
by extracted document type and then original filename before `DOC-###` IDs and
claim-global page numbers are assigned.

## What It Does

1. Runs Azure Document Intelligence OCR with `prebuilt-layout`.
2. For a combined PDF, detects page boundaries and writes split logical
   documents. For a folder, classifies and sorts the complete PDFs without
   splitting them.
3. Chunks OCR text by document and page range, assigning stable page IDs and
   chunk citation references.
4. Extracts metadata for each logical document:
   `id`, `title`, `summary`, `involved_parties`, `events`,
   `document_type`, and `page_range`.
   Event dates use nullable numeric `year`, `month`, and `day` fields, and every
   event cites a supporting chunk through `source_ref`.
5. In `work` mode, embeds each chunk with Snowflake Cortex `AI_EMBED`.
6. In `work` mode, stores chunk vectors in a claim-local Chroma vector store.
   In `home` mode, leaves embeddings empty and does not create an index.
7. Exposes ingestion, listing, search, and chunk-read functions through
   `claim_kb.api`.

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

  # snowflake mode only
  index/
    chroma/
```

Important output files:

- `source/claim.pdf`: preserved combined source PDF, when that input mode is used
- `documents/*.pdf`: split documents for combined input, or unchanged
  original-name PDFs for folder input
- `manifest.json`: claim manifest and one metadata record per logical document
- `pages.jsonl`: OCR text and stable page IDs such as `CLM-001:p1`
- `chunks.jsonl`: chunk text, optional embeddings, page IDs, and stable citation
  references such as `CLM-001/DOC-001#DOC-001-CHUNK-001`
- `index/chroma/`: local vector index for retrieval in `snowflake` mode only
- `run_log.json`: step-level ingestion status

`manifest.json` records the ordered `source_files`, documents, and
`embedding_mode`. In `snowflake` mode it also records the embedding provider,
model, and vector-store path. In `none` mode those fields are `null`.

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

folder_claim = kb.ingest_claim_folder(
    claim_id="PROP-B2B-2026-0417",
    folder_path="examples/claim_kb/sample_input",
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
    ingest_claim_folder,
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
ingest_claim_pdf(
    claim_id: str,
    pdf_path: str,
) -> StructuredClaimFile

ingest_claim_folder(
    claim_id: str,
    folder_path: str,
) -> StructuredClaimFile

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

`search_claim_file()` reads the persisted `embedding_mode`: it uses Snowflake
query embeddings and Chroma for `snowflake` claims, and the same local keyword
ranking as the lexical knowledge store for `none` claims. Keyword scores are
raw matched-term occurrence counts; equal scores retain chunk order.

## Configuration

Azure identity and OpenAI-compatible client construction are shared through
`knowledge_agent.infrastructure`; Claim KB keeps only its claim-specific OCR,
classification, ingestion, and persistence behavior.

`KNOWLEDGE_AGENT_MODE` is required and accepts only `home` or `work`.
The former `LLM_PROVIDER`, `LLM_MODEL`, and `--embedding-mode` switches are not
supported; the selected mode owns those decisions.

Home mode requires:

- `OPENROUTER_MODEL`
- `OPENROUTER_API_KEY`
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_API_KEY`

Work mode requires:

- `AZURE_OPENAI_MODEL`
- `AZURE_AI_PROJECT_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME`

The named project connection supplies the Document Intelligence endpoint only.
One `InteractiveBrowserCredential` authenticates both Azure AI Projects and the
Document Intelligence client. The connection target must be a custom-subdomain
endpoint that supports Microsoft Entra authentication.

Work embeddings are generated through Snowflake Cortex `AI_EMBED`; the code
creates a Snowpark session from the local Snowflake TOML connection config.
Home does not create Snowflake or Chroma clients.

Optional:

- `LLM_REASONING_EFFORT`: `low`, `medium`, or `high`; defaults to `medium`
- `SNOWFLAKE_CONNECTION_NAME`: local Snowflake connection name, defaults to
  `default`
- `SNOWFLAKE_EMBEDDING_MODEL`: Snowflake `AI_EMBED` model, defaults to
  `snowflake-arctic-embed-l-v2.0`
- `CLAIM_KB_DATA_ROOT`: output root, defaults to `data/claims`

Example home configuration:

```powershell
$env:KNOWLEDGE_AGENT_MODE="home"
$env:OPENROUTER_MODEL="provider/model"
$env:OPENROUTER_API_KEY="..."
$env:AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://my-doc-intel.cognitiveservices.azure.com"
$env:AZURE_DOCUMENT_INTELLIGENCE_API_KEY="..."
```

Example work configuration:

```powershell
$env:KNOWLEDGE_AGENT_MODE="work"
$env:AZURE_OPENAI_MODEL="my-azure-deployment"
$env:AZURE_AI_PROJECT_ENDPOINT="https://example.services.ai.azure.com/api/projects/my-project"
$env:AZURE_DOCUMENT_INTELLIGENCE_CONNECTION_NAME="document-intelligence"
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

The default tests use mocked provider and Snowflake clients plus generated
single-PDF and folder inputs. Live OpenRouter and Azure tests are separately
marked and opt-in.
