# Claim Researcher Agent

This package contains the claim research agent used by the CLI and Streamlit
chat.

## Main entry points

- `run_claim_research(...)` runs the bounded, cited research workflow.
- `plan_research(...)`, `extract_findings(...)`, `review_gaps(...)`, and
  `write_answer(...)` are plain LLM-stage functions that receive an explicit
  structured-output parser callable.
- Pydantic models in `models.py` define the workflow contracts.

## Responsibilities

- Plan focused searches over one persisted claim.
- Run independent `claim_search` tool calls with up to four worker threads.
- Extract and validate source-backed findings.
- Review gaps between layers.
- Write a final answer using validated findings only.

Concurrent work is committed in planned-query order, and progress callbacks run
on the main thread. Provider-specific structured-output mechanics stay in
`llm/`.
