# Knowledge Agent

Proof-of-concept backend for turning insurance claim documents into structured,
searchable evidence and producing cited research answers.

## Structure

```text
knowledge_agent/
  config.py
  agents/    # classifier/researcher prompts, models, tools, workflows
  llm/       # provider configuration and structured-output calls
  claims/    # ingestion, persistence, lexical/semantic retrieval
  research/  # research CLI
```

AI behavior lives under `agents/`: the document classifier supports ingestion,
and the claim researcher supports cited answers. The shared `llm` package
contains provider setup only; claim and research prompts stay in their owning
agent packages.

## Runtime profiles

`KNOWLEDGE_AGENT_PROFILE` selects one explicit dependency bundle:

- `api_key`: NVIDIA DeepSeek V4 Pro, API-key Document Intelligence, and lexical
  retrieval.
- `azure_project`: Azure AI Projects with browser authentication, a named
  Document Intelligence connection, Snowflake Cortex embeddings, and Chroma
  semantic retrieval.

Copy `.env.example` to `.env` and fill in the selected profile's values.

## Ingest a claim

```powershell
python -m knowledge_agent.claims.cli `
  --claim-id CLM-001 `
  --folder-path examples/claims/sample_input
```

Use `--pdf-path` instead when one PDF contains several logical documents. Output
is written under `CLAIM_DATA_ROOT` (`data/claims` by default). See
[`knowledge_agent/claims/README.md`](knowledge_agent/claims/README.md) for the
persisted format and programmatic entry points.

## Research a claim

```powershell
python -m knowledge_agent.research.cli `
  --claim-path examples/claims/sample_output `
  --question "What repairs were invoiced?"
```

Research opens the retrieval strategy recorded in `manifest.json`: lexical
claims remain fully local, while semantic claims create a Snowflake query
embedder and read their claim-local Chroma index. Both paths return identical
page IDs and `source_ref` citations to the research loop.

Research logs append to `logs/research.log`. `DEBUG` logging includes prompts,
claim evidence, findings, and answers.

## Claim Research Workbench

The Streamlit workbench combines ingestion, knowledge-base inspection, and a
context-aware research chat:

```powershell
streamlit run knowledge_agent/app.py
```

Select an existing claim from `CLAIM_DATA_ROOT`, or ingest either one combined
claim PDF or several already-separated document PDFs from the sidebar. The
knowledge-base tab separates the aggregate timeline, claim-wide party list,
and document workspace. The Documents view summarizes every file and its
metadata, then organizes the selected document's metadata, evidence chunks,
OCR page text, and exact source references into focused tabs. The chat tab
researches only the selected claim, keeps its conversation history for the
current browser session, annotates answers with source tooltips, and exposes
the agent's steps and retrieval tool calls. An off-by-default Audit mode records
every Research Agent system/user prompt and parsed result plus full retrieval
inputs and evidence. Each captured turn has its own audit expander, including
partial traces for failed runs.

Audit traces can repeat complete claim evidence and conversation text. They are
kept only in the current Streamlit session, are hidden when Audit mode is off,
and are removed with the selected claim's chat history. They are not written to
the claim directory.

The app uses the active `KNOWLEDGE_AGENT_PROFILE` and the same `.env` provider
configuration as the CLIs. Credentials remain server-side and are never entered
through the UI. Existing claim IDs are not overwritten.

## Tests

```powershell
python -m pytest
```

Live provider contracts remain opt-in through `RUN_NVIDIA_CONTRACT_TEST=1`
or `RUN_AZURE_CONTRACT_TEST=1`.

## Paid API-key ingestion runs

Two live ingestion tests exercise the complete `api_key` profile with retained
debug artifacts. They are skipped by normal `pytest` runs and require separate
shell flags; do not add these flags to `.env`.

The small run processes two two-page documents. It makes two Azure Document
Intelligence requests and approximately four NVIDIA requests:

```powershell
$env:RUN_API_KEY_SMALL_INGESTION="1"
python -m pytest tests/live/test_api_key_ingestion.py `
  -m live_api_key_ingestion -k small -s
Remove-Item Env:RUN_API_KEY_SMALL_INGESTION
```

The full run processes all fourteen sample documents and twenty-seven pages. It
makes fourteen Document Intelligence requests and approximately twenty-eight
NVIDIA requests:

```powershell
$env:RUN_API_KEY_FULL_INGESTION="1"
python -m pytest tests/live/test_api_key_ingestion.py `
  -m live_api_key_ingestion -k full -s
Remove-Item Env:RUN_API_KEY_FULL_INGESTION
```

Outputs remain under `data/live-runs/API-KEY-SMALL/` or
`data/live-runs/API-KEY-FULL/`. Each folder includes `debug.log`, which contains
OCR-derived claim text, classifier prompts, parsed model output, request IDs,
token usage, and latency. These folders are ignored by Git.
