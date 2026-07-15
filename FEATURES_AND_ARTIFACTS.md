# Features and Artifacts

This document describes the claim knowledge agent from a product and data
perspective: what it does for a claim and the artifacts it creates, maintains,
or returns.

## Product features

### Claim ingestion

The system turns either one combined claim PDF or a folder of already-separated
claim PDFs into a self-contained claim knowledge base. It preserves the source
documents, captures OCR text page by page, organizes logical documents, and
extracts document metadata such as titles, types, parties, events, and concise
summaries. Document summaries are limited to 200 words.

### Claim knowledge-base inspection

A stored claim can be explored through its documents, summaries, parties,
timeline events, evidence chunks, OCR page text, and source references. Every
event and every research citation can be traced back to a stored evidence
chunk and its source pages.

### Cited claim research

Users can ask questions about one stored claim. A bounded deep agent plans with
todos, iterates over claim-local searches, and returns a concise answer with
validated citations. Streamlit planning mode can clarify scope, pause for plan
approval, and then resume the same native model context. Each submitted question
is independent from earlier reports.

### Retrieval

Claims support lexical retrieval over stored text, semantic retrieval over a
claim-local Chroma index, or embedded LightRAG retrieval over a claim-local
entity graph and vector index. All modes return the same citation-bearing chunk
evidence to the research agent. A claim can retain Custom and LightRAG indexes
at the same time; Streamlit selects one for each new report and records it in the
audit snapshot. LightRAG never generates the final report.

## Claim knowledge-base layout

Each ingested claim is stored at `CLAIM_DATA_ROOT/<claim_id>/` (`data/claims`
by default).

```text
<claim_id>/
  source/claim.pdf       # original combined input, when applicable
  documents/*.pdf        # logical documents used by the claim
  manifest.json          # claim catalog and document metadata
  pages.jsonl            # OCR text, one page per line
  chunks.jsonl           # retrievable evidence, one chunk per line
  run_log.json           # ingestion status and timing
  research/history.json  # plans, reports, native messages, audit, and usage
  index/chroma/          # semantic retrieval only
  index/lightrag/        # LightRAG graph, vectors, KV data, and metadata
```

`claim_id` identifies the claim directory and must be non-empty. It cannot be
`.` or `..`, and cannot contain `/` or `\`. The claim directory is the unit of
storage, retrieval, citation, and research scope.

## Naming and citation conventions

| Item | Convention | Example |
| --- | --- | --- |
| Logical document ID | `DOC-<three-digit sequence>` | `DOC-002` |
| Page ID | `<claim_id>:p<claim-wide page number>` | `CLM-001:p3` |
| Chunk ID | `<document_id>-CHUNK-<three-digit sequence>` | `DOC-002-CHUNK-001` |
| Source reference | `<claim_id>/<document_id>#<chunk_id>` | `CLM-001/DOC-002#DOC-002-CHUNK-001` |
| Split-PDF filename | `<document_id>_<document-type-slug>.pdf` | `DOC-002_repair_invoice.pdf` |

Document and chunk sequences are zero-padded to three digits. A document-type
slug is lowercase, with runs of non-alphanumeric characters replaced by `_`.
Page numbers are claim-wide: folder inputs are renumbered after the documents
are placed in their deterministic claim order.

## Artifacts

### Source and logical-document PDFs

**Purpose:** retain the PDF material behind the claim knowledge base.

- `source/claim.pdf` exists only when the input was one combined PDF. It is the
  preserved original.
- `documents/*.pdf` contains the claim's logical documents. For a combined
  input, the original is split into the `DOC-<nnn>_<type>.pdf` names above. For
  a folder input, each supplied PDF is copied under its original filename.
- `manifest.json` records these files as relative `source_files` paths and
  links each logical document to its `file_name`.

Example combined-PDF output:

```text
source/claim.pdf
documents/DOC-001_fnol.pdf
documents/DOC-002_invoice.pdf
```

### `manifest.json`

**Purpose:** the claim-level catalog. It identifies the claim, lists its source
files, and holds the document metadata used for inspection and research.

**Expected structure:** one JSON object with these important fields.

| Field | Meaning |
| --- | --- |
| `claim_id` | The ID of this claim and its artifact directory. |
| `source_files` | Relative paths to the preserved input or document PDFs. |
| `documents` | Ordered logical-document metadata records. |
| `chunk_count` | Total number of records in `chunks.jsonl`. |
| `retrieval_mode` | `lexical`, `semantic`, or `lightrag`. |
| `additional_retrieval_modes` | Other indexes available for explicit per-run selection; `both` adds `lightrag` to a Custom primary mode. |
| `embedding_provider`, `embedding_model` | Present for semantic and LightRAG claims; `null` for lexical claims. |
| `created_at` | UTC creation timestamp. |

Each entry in `documents` contains:

| Field | Meaning |
| --- | --- |
| `id` | Stable logical document ID, such as `DOC-001`. |
| `title`, `document_type` | Human-readable document identity and type. |
| `summary` | Concise document summary, no more than 200 words. |
| `page_range` | Inclusive `start_page` and `end_page` in claim-wide numbering. |
| `file_name` | Filename under `documents/`. |
| `involved_parties` | Party records with non-empty `name` and `role`. |
| `events` | Event records with `year`, `month`, `day`, `sentence`, and `source_ref`. Date parts may be `null` when unknown. |

Every event's `source_ref` must identify a chunk belonging to the same
document. For example:

```json
{
  "id": "DOC-002",
  "title": "Repair Invoice",
  "summary": "Invoice for repair labor and parts.",
  "document_type": "invoice",
  "page_range": {"start_page": 2, "end_page": 2},
  "file_name": "DOC-002_invoice.pdf",
  "involved_parties": [{"name": "Example Body Shop", "role": "repair vendor"}],
  "events": [
    {
      "year": 2026,
      "month": 6,
      "day": 3,
      "sentence": "Example Body Shop issued the repair invoice.",
      "source_ref": "CLM-001/DOC-002#DOC-002-CHUNK-001"
    }
  ]
}
```

### `pages.jsonl`

**Purpose:** preserve the OCR text for every claim page and provide the page
records shown during claim inspection.

**Expected structure:** one JSON object per line, in claim-wide page order.

| Field | Meaning |
| --- | --- |
| `claim_id` | Owning claim ID. |
| `page_number` | One-based, claim-wide page number. |
| `page_id` | Stable ID derived from the claim ID and page number. |
| `text` | OCR text for that page; it may be empty when no text was extracted. |

Example line:

```json
{"claim_id":"CLM-001","page_number":3,"page_id":"CLM-001:p3","text":"Repair Invoice\\nTotal: 850.00"}
```

### `chunks.jsonl`

**Purpose:** store the retrievable evidence used by claim search and cited
research. A chunk represents one non-empty span of OCR text from a logical
document and records exactly which pages support it.

**Expected structure:** one JSON object per line, in deterministic chunk order.

| Field | Meaning |
| --- | --- |
| `claim_id`, `document_id` | Claim and logical document that own the chunk. |
| `chunk_id` | Stable chunk identifier within the document. |
| `source_ref` | Citation-ready reference for this exact chunk. |
| `chunk_index` | Zero-based position across all claim chunks. |
| `document_type` | Type of the owning document. |
| `page_range`, `page_ids` | Inclusive page span and its ordered, unique page IDs. |
| `text` | The evidence text used for retrieval and research. |

Example line:

```json
{"claim_id":"CLM-001","document_id":"DOC-002","chunk_id":"DOC-002-CHUNK-001","source_ref":"CLM-001/DOC-002#DOC-002-CHUNK-001","chunk_index":1,"document_type":"invoice","page_range":{"start_page":3,"end_page":3},"page_ids":["CLM-001:p3"],"text":"Page 3\\nRepair Invoice\\nTotal: 850.00"}
```

The `source_ref` is the durable bridge between metadata events, search results,
and citations in the final answer.

### `index/chroma/` (claims with semantic retrieval)

**Purpose:** keep the claim-local semantic retrieval index.

**Expected structure:** an index-owned directory at `index/chroma/`. Its
internal files are not a public artifact format; use `manifest.json` to decide
whether it is required and `chunks.jsonl` for the portable evidence records.
The index associates vectors and retrieval metadata with `chunk_id` values. It
is absent for lexical claims, whose retrieval relies on `chunks.jsonl` and
`manifest.json` only.

### `index/lightrag/` (claims with LightRAG retrieval)

**Purpose:** keep the embedded LightRAG JSON KV, NanoVectorDB, NetworkX graph,
document-status data, LLM cache, and index metadata for one claim.

`metadata.json` records the LightRAG version, indexing LLM, embedding
provider/model/dimension/context, indexed chunk count, graph counts, indexing
usage, and creation time. Each claim chunk is inserted with its stable
`source_ref` as document ID and its `chunk_id` as the retrieval file path.
Unknown paths returned by LightRAG fail the search rather than producing a
partial result.

The index can be rebuilt atomically from `chunks.jsonl`. A successful rebuild
replaces only the retrieval index and manifest retrieval fields. A failed
rebuild leaves the previous engine intact. Validation also checks LightRAG's
document-status store, so a graph with failed or missing vector flushes is not
treated as a complete knowledge base.

### `run_log.json`

**Purpose:** record the progress and outcome of an ingestion attempt. It is
updated as the claim is processed, so it can also describe a failed run.

**Expected structure:** one JSON object with:

| Field | Meaning |
| --- | --- |
| `claim_id` | Claim being ingested. |
| `entries` | Ordered ingestion-status entries. |
| `created_at`, `finished_at` | UTC timestamps for the overall run; `finished_at` can be `null` while active. |

Each entry has `step`, `status`, `message`, `started_at`, and `finished_at`.
`status` is recorded as `running`, `succeeded`, or `failed`; a failed entry
includes the surfaced error in `message`.

### Research answer and audit history

**Purpose:** represent the answer to one question over one claim, together with
the evidence trail that supports it.

This result is returned to the caller and Streamlit persists it in
`research/history.json` together with its plan, native messages, tool evidence,
usage, and knowledge-base snapshot. Its important fields are:

| Field | Meaning |
| --- | --- |
| `answer` | The rendered answer text. |
| `source_refs` | Exact source references declared by the answer. |
| `evidence_sufficient` | Whether retrieved claim evidence supports an answer. |
| Native messages | PydanticAI model/tool messages used for chat history and traces. |
| Native usage | Model-request, tool-call, and token totals for the turn. |

Factual statements in `answer` use square-bracket citations, for example:

```text
The repair invoice lists a total of 850.00 [CLM-001/DOC-002#DOC-002-CHUNK-001].
```

### Operational logs

**Purpose:** retain diagnostic records for ingestion and research activity.

- `logs/claims.log` records claim-ingestion activity.
- `logs/research.log` records claim-research activity.

These are append-only operational artifacts rather than part of an individual
claim's portable knowledge base. Debug logging can contain OCR-derived text,
prompts, model output, request identifiers, and timing information. Treat these
files, as well as runtime claim directories, as sensitive claim data and keep
them out of version control.

## Artifact relationships

```text
PDF sources
  -> pages.jsonl
  -> chunks.jsonl
  -> manifest.json (documents, parties, events)

chunks.jsonl
  -> index/chroma/ when semantic retrieval is available
  -> index/lightrag/ when LightRAG retrieval is available

manifest events + chunks
  -> source_ref citations
  -> persisted cited research reports

ingestion activity
  -> run_log.json and operational logs
```
