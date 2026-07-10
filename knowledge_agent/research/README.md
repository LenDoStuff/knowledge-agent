# Research

The research package owns the command-line entry point for claim research. The
canonical implementation lives in `knowledge_agent.agents.claim_researcher`.

## Main entry points

- `knowledge_agent.agents.claim_researcher.run_claim_research(...)` runs the
  typed research loop over an open `ClaimStore`.
- `knowledge_agent.llm.client.open_structured_output_parser(settings)` yields
  the parser callable passed into the research workflow.
- `python -m knowledge_agent.research.cli` runs the same flow from the command
  line.

## Flow

Research stays sequential where decisions depend on prior state:

1. Plan objectives and initial searches.
2. For each layer, run independent search-and-finding tasks with up to four
   worker threads.
3. Validate finding source references against retrieved evidence.
4. Review gaps and optionally plan the next layer.
5. Write the final answer from validated findings only.

Concurrent query results are committed in planned-query order, and progress
callbacks run on the main thread.

## Constraints

Research reads from a claim store but does not mutate the persisted claim
knowledge base. Answers should cite stable claim `source_ref` values only.
Unsupported or malformed model output should fail explicitly rather than being
silently repaired.
