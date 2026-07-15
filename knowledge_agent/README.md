# knowledge_agent

This package is the application boundary for the proof-of-concept claim
knowledge agent. It wires together environment configuration, the Streamlit
workbench, claim ingestion/retrieval, shared LLM clients, and the AI agents
used for classification and cited research.

## What belongs here

- `config.py` loads the deployment profile and validates simple environment
  values shared by the app.
- `app.py` composes the Streamlit page and sidebar.
- `ui/` owns upload handling, claim views, and research interaction rendering.
- `agents/` owns AI prompts, structured contracts, tools, validation, and
  PydanticAI agent code.
- `claims/` owns persisted claim knowledge bases.
- `llm/` owns PydanticAI provider setup and runtime resources.
- `research/` owns research-history persistence and the research CLI.

Dependencies should stay straightforward: agent packages may use `claims` and
`llm`; `claims` composes the document-classifier agent for ingestion; `llm`
should not know about claim or research behavior.

## Runtime shape

The active deployment profile comes from `KNOWLEDGE_AGENT_PROFILE`:

- `api_key` uses NVIDIA DeepSeek V4 Pro through PydanticAI and lexical custom
  retrieval for claims.
- `azure_project` uses PydanticAI with Azure AI Projects, Document Intelligence,
  Snowflake Cortex embeddings, and Chroma semantic retrieval.

Both profiles can create the Custom engine, embedded LightRAG, or both. Each
index uses the existing claim chunks and stable citation IDs. When both exist,
each Streamlit research interaction explicitly selects one engine and persists
that choice in its audit snapshot.

The Streamlit app and CLIs use the same `.env`-driven configuration. Keep
credentials server-side; the UI should not ask users to paste secrets.
Research answers render only when every citation resolves to evidence in the
selected claim. Missing or stale citation references surface as explicit UI
errors instead of partially rendered answers. Claim documents are grouped into
inventory, metadata, evidence, and OCR views. Full research audit data is always
persisted, and the UI warns that native PydanticAI messages and events can
contain full claim and conversation text.
