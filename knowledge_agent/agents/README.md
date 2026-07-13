# Agents

This package owns AI behavior that is more specific than provider setup:
prompts, structured model contracts, tools, and validation.

## Packages

- `document_classifier/` classifies claim PDFs, page boundaries, and document
  metadata during ingestion.
- `claim_researcher/` is a claim-scoped Pydantic Deep Agents researcher with
  clarification and approval planning, retrieval tools, native history, and
  citation validation.

PydanticAI provider models and runtime ownership stay in `llm/`.
Claim persistence, OCR, chunking, indexing, and retrieval stay in `claims/`.
