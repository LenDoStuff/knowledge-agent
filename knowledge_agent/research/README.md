# Research

This package owns the command-line entry point for claim research. The
canonical deep-agent implementation lives in
`knowledge_agent.agents.claim_researcher`.

## Main entry points

- `run_claim_research(runtime, store, question, ...)` returns a native
  PydanticAI result.
- `history.py` validates and atomically persists independent, auditable research
  interactions under the selected claim.
- `open_agent_runtime(settings)` supplies the configured provider model.
- `python -m knowledge_agent.research.cli` runs the same claim-scoped flow.

The CLI exposes `--top-k`, `--max-searches`, and `--request-limit`. Research
uses todo planning and iterative `claim_search` calls, then returns a structured
cited answer. The Streamlit workbench can first clarify scope and pause on a
structured plan. It persists messages, plans, reports, audit events, and usage;
claim evidence and citation identifiers are unchanged.

The CLI automatically opens the retrieval engine persisted in the claim
manifest. LightRAG is retrieval-only: its structured chunk results feed
`claim_search`, while Pydantic Deep remains responsible for planning, reporting,
citation validation, native history, and audit events.
