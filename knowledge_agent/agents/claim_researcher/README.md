# Claim Researcher Agent

This package contains the claim research agent used by the CLI and Streamlit
chat.

## Main entry points

- `run_claim_research(...)` runs the bounded, cited research workflow. Its
  optional `on_audit` callback receives typed LLM and retrieval audit entries.
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
- Emit an exact, ordered prompt/result and retrieval trace when a caller opts
  into auditing.

Concurrent work is committed in planned-query order, and progress and audit
callbacks run on the main thread. Query workers collect their audit entries
locally so one failed query does not hide calls completed elsewhere in the
same layer. Provider-specific structured-output mechanics stay in `llm/`.

Audit entries contain system and user prompts, parsed structured model results,
and complete `claim_search` evidence. They do not contain raw provider response
envelopes or hidden reasoning. Callers own retention; the workflow does not
write audit traces to the claim knowledge base.
