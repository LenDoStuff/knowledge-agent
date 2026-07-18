# Knowledge Agent

Proof-of-concept backend for turning insurance claim documents into structured,
searchable evidence and producing cited research answers.

## Structure

```text
knowledge_agent/
  config.py
  agents/    # classifier/researcher prompts, models, tools, workflows
  llm/       # PydanticAI provider configuration and runtime
  claims/    # ingestion, persistence, custom/LightRAG retrieval
  research/  # research history and CLI
  ui/        # Streamlit claim, knowledge-base, report, and research views
```

AI behavior lives under `agents/`: the document classifier supports ingestion,
and the Pydantic Deep Agents researcher supports cited answers. The shared
`llm` package contains PydanticAI provider setup only; claim and research
prompts stay in their owning agent packages.

## Runtime profiles

`KNOWLEDGE_AGENT_PROFILE` selects one explicit dependency bundle:

- `api_key`: NVIDIA DeepSeek V4 Pro, API-key Document Intelligence, and lexical
  retrieval.
- `azure_project`: Azure AI Projects with browser authentication, a named
  Document Intelligence connection, Snowflake Cortex embeddings, and Chroma
  semantic retrieval.
- `snowflake`: Snowflake Cortex REST through its OpenAI-compatible Chat
  Completions endpoint, `AI_PARSE_DOCUMENT` OCR, Snowflake Cortex embeddings,
  and Chroma semantic retrieval.

Either profile can build Custom, embedded LightRAG, or both knowledge bases from
the same persisted chunks. LightRAG reuses the selected profile's LLM through
PydanticAI; `api_key` uses NVIDIA `nvidia/llama-nemotron-embed-1b-v2`
embeddings; `azure_project` and `snowflake` use the configured Snowflake
embedding model.

Copy `.env.example` to `.env` and fill in the selected profile's values.

The Snowflake profile reads account, user, role, warehouse, database, schema,
and interactive authentication from a native named connection in
`~/.snowflake/connections.toml` (or Snowflake Connector's Windows location).
It requires `SNOWFLAKE_CORTEX_PAT` separately because the OpenAI-compatible REST
endpoint requires a bearer token. The configured role must have
`SNOWFLAKE.CORTEX_USER`, database/schema usage, and `CREATE STAGE` on the schema.
The application creates `SNOWFLAKE_DOCUMENT_STAGE`, uploads each OCR input under
a unique prefix, and removes that prefix after parsing; the empty managed stage
remains available for later runs.

```toml
# ~/.snowflake/connections.toml
[knowledge_agent]
account = "org-account"
user = "personal-user"
authenticator = "externalbrowser"
client_store_temporary_credential = true
warehouse = "COMPUTE_WH"
role = "KNOWLEDGE_AGENT_ROLE"
database = "KNOWLEDGE_AGENT_DB"
schema = "PUBLIC"
```

```dotenv
KNOWLEDGE_AGENT_PROFILE=snowflake
SNOWFLAKE_CONNECTION_NAME=knowledge_agent
SNOWFLAKE_CORTEX_PAT=<programmatic-access-token>
SNOWFLAKE_CORTEX_MODEL=<tool-capable-cortex-model>
SNOWFLAKE_DOCUMENT_STAGE=KNOWLEDGE_AGENT_DOCUMENTS
```

## Ingest a claim

```powershell
python -m knowledge_agent.claims.cli `
  --claim-id CLM-001 `
  --folder-path examples/claims/sample_input `
  --knowledge-base both
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

The CLI opens the default retrieval strategy recorded in `manifest.json`:
lexical claims remain fully local, semantic claims use Snowflake/Chroma, and
LightRAG claims use their embedded graph and vector stores. Streamlit lets the
user choose Custom or LightRAG for each new report when both are present. All
paths return identical page IDs and `source_ref` citations.

Research logs append to `logs/research.log`. `DEBUG` logging includes prompts,
claim-search evidence, native usage, and answers.

## Claim Research Workbench

The Streamlit workbench combines ingestion, knowledge-base inspection, and a
context-aware research chat:

```powershell
streamlit run knowledge_agent/app.py
```

Select an existing claim from `CLAIM_DATA_ROOT`, or ingest either one combined
claim PDF or several already-separated document PDFs from the sidebar. The
knowledge-base tab shows document metadata, parties, events, evidence chunks,
OCR page text, an aggregate timeline, a claim-wide party list, and exact source
references. LightRAG indexes also expose searchable entity and relationship
tables. A protected rebuild action can create Custom, LightRAG, or both from
persisted chunks without rerunning OCR. The chat tab researches only the selected
claim and selected engine, keeps its
independent report history under that claim, annotates answers with
source tooltips, and exposes native retrieval tool calls and model usage.
Planning mode clarifies scope and pauses for approval before searching. Reports,
plans, native PydanticAI messages, streamed events, tool inputs/results, and
usage persist under the claim for later audit in the Report history view.

The app uses the active `KNOWLEDGE_AGENT_PROFILE` and the same `.env` provider
configuration as the CLIs. Credentials remain server-side and are never entered
through the UI. Existing claim IDs are not overwritten.

## Tests

```powershell
python -m pytest
```

Live provider contracts remain opt-in through `RUN_NVIDIA_CONTRACT_TEST=1`,
`RUN_AZURE_CONTRACT_TEST=1`, or `RUN_SNOWFLAKE_CONTRACT_TEST=1`.

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
