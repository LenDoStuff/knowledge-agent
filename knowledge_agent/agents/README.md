# Agents

This package owns AI behavior that is more specific than provider setup:
prompts, structured model contracts, workflow state, tools, and validation.

## Packages

- `document_classifier/` classifies claim PDFs, page boundaries, and document
  metadata during ingestion.
- `claim_researcher/` plans claim searches, extracts findings, validates
  citations, and writes grounded answers.

Provider clients and structured-output request mechanics stay in `llm/`.
Claim persistence, OCR, chunking, indexing, and retrieval stay in `claims/`.
