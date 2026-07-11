# knowledge_agent

This package is the application boundary for the proof-of-concept claim
knowledge agent. It wires together environment configuration, the Streamlit
workbench, claim ingestion/retrieval, shared LLM clients, and the AI agents
used for classification and cited research.

## What belongs here

- `config.py` loads the deployment profile and validates simple environment
  values shared by the app.
- `app.py` is the Streamlit workbench for upload, ingestion, claim inspection,
  and research chat.
- `agents/` owns AI prompts, structured contracts, tools, validation, and
  workflow code.
- `claims/` owns persisted claim knowledge bases.
- `llm/` owns provider setup and structured-output requests.
- `research/` owns the research CLI.

Dependencies should stay straightforward: agent packages may use `claims` and
`llm`; `claims` composes the document-classifier agent for ingestion; `llm`
should not know about claim or research behavior.

## Runtime shape

The active deployment profile comes from `KNOWLEDGE_AGENT_PROFILE`:

- `api_key` uses NVIDIA DeepSeek V4 Pro for structured LLM calls and lexical
  retrieval for claims.
- `azure_project` uses Azure AI Projects, Azure Document Intelligence,
  Snowflake Cortex embeddings, and Chroma semantic retrieval.

The Streamlit app and CLIs use the same `.env`-driven configuration. Keep
credentials server-side; the UI should not ask users to paste secrets.
Research answers render only when every citation resolves to evidence in the
selected claim. Missing or stale citation references surface as explicit UI
errors instead of partially rendered answers.

The research chat also owns an off-by-default, session-wide Audit mode. When it
is enabled before a turn, the app retains that turn's exact Research Agent
prompts, parsed outputs, retrieval inputs, and full evidence in Streamlit
session state. Successful and failed traces are never persisted to a claim
folder and are removed when that claim's chat is cleared.
