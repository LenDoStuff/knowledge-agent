# Claim KB Example Output

This folder contains hand-written synthetic examples of `claim_kb` output. The
files are documentation artifacts: they show the shape of persisted metadata,
OCR pages, chunks, and logs without requiring Azure, Snowflake, Chroma, or a
real claim PDF.

The sample uses fake claim ID `CLM-SAMPLE-001` and two fake documents:

- `DOC-001`: first notice of loss (`fnol`)
- `DOC-002`: repair invoice (`invoice`)

Document metadata includes party roles and event sentences. Event dates use
nullable numeric `year`, `month`, and `day` fields so partial dates can stay
partial. Every event has a `source_ref` pointing to its supporting chunk.

Pages use stable IDs such as `CLM-SAMPLE-001:p1`. Chunks list their source page
IDs and use citation references such as
`CLM-SAMPLE-001/DOC-001#DOC-001-CHUNK-001`.

The embeddings in `sample_output/chunks.jsonl` are tiny deterministic
vectors only to show the schema shape. They are not real model output and should
not be used for similarity search quality checks.

Runtime ingestion output still belongs under `data/claims/<claim_id>/`, which is
ignored by Git. Do not put real claim data, generated PDFs, Chroma files, or
customer-derived examples in this folder.

## Files

```text
examples/claim_kb/
  README.md
  sample_output/
    manifest.json
    pages.jsonl
    chunks.jsonl
    run_log.json
```

The example intentionally omits `source/`, `documents/`, and `index/` contents.
Those are binary or generated runtime artifacts, not useful committed
documentation fixtures for this proof of concept.
