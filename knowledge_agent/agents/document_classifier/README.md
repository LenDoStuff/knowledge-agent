# Document Classifier Agent

This package contains the LLM-assisted document classifier used by claim
ingestion.

## Main entry points

- `classify_document(...)`, `classify_page_boundary(...)`, and
  `extract_document_metadata(...)` are plain functions that receive an
  explicit structured-output parser callable.
- `LogicalDocument`, `PageBoundaryDecision`, `DocumentClassification`, and
  `ExtractedDocumentMetadata` define the structured model contracts.

## Responsibilities

- Classify already-separated uploaded PDFs.
- Decide page boundaries for a combined scanned claim PDF.
- Extract document metadata from claim chunks.

The metadata prompt must keep summaries to no more than 200 words. Prompt text
should stay here; provider-specific JSON mode belongs in `llm/`.
