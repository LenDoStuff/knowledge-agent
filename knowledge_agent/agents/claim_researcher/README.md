# Claim Researcher Agent

This package contains the claim-scoped Pydantic Deep Agents researcher used by
the CLI and Streamlit chat.

## Main entry points

- `run_claim_planning(...)` returns a structured clarification question or
  `ClaimResearchPlan` with native messages and usage.
- `run_claim_research(...)` returns
  `AgentRunResult[ClaimResearchOutput]` with native messages and usage.
- `ClaimResearchOutput` contains the cited answer, ordered source references,
  and whether the retrieved evidence was sufficient.

## Behavior

- The optional planning phase uses todo tools, allows at most three
  clarification rounds, and has no retrieval capability.
- Approved plans and their native message history are passed into the research
  phase so clarification, approval, searches, and reporting remain one context.
- `claim_search(query, research_goal)` is the only domain tool. It asynchronously
  dispatches to custom or LightRAG retrieval and returns the same citation-rich
  evidence shape.
- A turn is limited to six searches and ten model requests by default. For a
  LightRAG claim, query keyword extraction also runs through PydanticAI and is
  included in this request budget and native usage total.
- Search calls from one model response may run concurrently.
- Output validation accepts only exact source references returned during the
  current run and requires inline citations to match `source_refs` in order.
- Native PydanticAI messages carry chat context and drive the workbench trace.
- A safe sliding window keeps the latest 20 messages after history reaches 40,
  without making unbudgeted summarizer requests.

Filesystem, shell, web, persistent memory, skills, generic subagents, teams,
and checkpoints are explicitly disabled. The agents never mutate claim evidence.
