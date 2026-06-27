# Claim KB Examples

This folder contains synthetic input PDFs and hand-written `ingest` output.
`sample_input/` demonstrates folder ingestion with separate claim documents.
`sample_output/` shows persisted metadata, OCR pages, chunks, and logs without
requiring Azure, Snowflake, or Chroma. Research-agent tests load the output as
read-only input.

The folder input contains fourteen numbered PDFs. Folder ingestion OCRs each
top-level PDF as one document, classifies it, sorts documents by type and
filename, and copies it unchanged under the generated claim's `documents/`
folder. It does not split these PDFs.

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
ignored by Git. Do not put real claim data, Chroma files, or customer-derived
examples in this folder.

## Files

```text
examples/ingest/
  README.md
  sample_input/
    00_claim_file_index.pdf
    01_fnol_and_broker_notice.pdf
    ...
    13_coverage_reservation_memo.pdf
  sample_output/
    manifest.json
    pages.jsonl
    chunks.jsonl
    run_log.json
```

The output example intentionally omits `source/`, `documents/`, and `index/`
contents. Those are runtime artifacts rather than useful output-shape fixtures.
