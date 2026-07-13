# Document Classifier Agent

This package contains the PydanticAI structured agents used by claim ingestion.

## Main entry points

- `classify_document(runtime, ...)` classifies an already-separated PDF.
- `classify_page_boundary(runtime, ...)` detects logical-document boundaries.
- `extract_document_metadata(runtime, ...)` extracts validated claim metadata.
- `LogicalDocument`, `PageBoundaryDecision`, `DocumentClassification`, and
  `ExtractedDocumentMetadata` define the structured output contracts.

The metadata prompt limits summaries to 200 words. Extracted event references
must exactly match chunks from the same logical document. Provider setup and
model transport remain in `knowledge_agent.llm`.
