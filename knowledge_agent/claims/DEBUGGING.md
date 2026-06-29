# Debugging the Small API-Key Ingestion

This path is for learning the ingestion flow without stepping through pytest.
It makes paid Azure Document Intelligence and NVIDIA requests for two
two-page sample documents.

## Start the run

1. Open **Run and Debug** in VS Code.
2. Select **PAID: Learn small API-key ingestion**.
3. Add the breakpoints listed below.
4. Press **F5** and type `1` when VS Code asks you to enable the paid run.

The launch reads credentials from `.env` and stops immediately unless
`KNOWLEDGE_AGENT_PROFILE=api_key`. It resets only
`data/live-runs/API-KEY-SMALL/`. Output and the detailed `debug.log` remain in
that directory after the run.

## Follow the important path

Set breakpoints at these function definitions before pressing F5:

1. `scripts/debug_small_claim_ingestion.py` — `main`
2. `knowledge_agent/claims/dependencies.py` — `live_ingestion_services`
3. `knowledge_agent/claims/pipeline.py` — `ingest_claim_folder`
4. `knowledge_agent/claims/ocr.py` — `extract_pages`
5. `knowledge_agent/claims/classify.py` — `classify_document`
6. `knowledge_agent/claims/pipeline.py` — `_complete_ingestion`

This is the useful mental model:

```text
debug entry
  -> build API-key services
  -> OCR each PDF
  -> classify each document
  -> prepare and renumber pages
  -> chunk documents
  -> extract metadata
  -> write pages, chunks, manifest, and run log
```

Use **F10** to execute the current line and stay in the current function. Use
**F11** only when the next call is one of the project functions above. If F11
does not enter useful code, press **F5** to continue to the breakpoint already
set inside the destination function. This is usually clearer than stepping
through context-manager, HTTP-client, or SDK machinery.

Useful variables to watch are `pdf_paths`, `pages`, `classification`,
`classified_documents`, `logical_documents`, `chunks`, `documents`, and
`manifest`.

## Ignore these paths for this run

- `embeddings.py`, `vector_store.py`, and Snowflake: the API-key profile is
  lexical and never uses them.
- `store.py` and `open_claim_store`: they retrieve an already-ingested claim.
- `split.py`, `group_logical_documents`, and `write_split_pdfs`: those are for
  one combined PDF, not the selected folder ingestion.
- pytest internals and fixtures: the direct debug launch does not load them.
- Azure, OpenAI, HTTPX, and authentication internals: use project breakpoints
  before and after these external calls.

The opt-in live tests remain the verification path. Run them separately when
you want assertions around the real provider calls; ordinary `pytest` skips
them.
