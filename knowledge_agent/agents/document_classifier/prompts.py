"""Prompt builders for the document classifier agent."""

from __future__ import annotations

from knowledge_agent.agents.document_classifier.models import LogicalDocument
from knowledge_agent.claims.models import DocumentChunk, PageText


def build_classify_document_prompt(
    file_name: str,
    pages: list[PageText],
) -> tuple[str, str]:
    document_text = "\n\n".join(
        f"Page {page.page_number}\n{page.text}" for page in pages
    )
    return (
        (
            "You classify complete documents in insurance claim files. "
            "Choose a concise title and a plain, specific document type."
        ),
        (
            "Classify this complete document.\n\n"
            f"File name: {file_name}\n\n"
            f"OCR text:\n{_clip(document_text, 10000)}"
        ),
    )


def build_page_boundary_prompt(
    page: PageText,
    prior_page: PageText,
    current_document: LogicalDocument | None,
) -> tuple[str, str]:
    current_context = "No current document."
    if current_document is not None:
        current_context = (
            f"Current document id: {current_document.id}\n"
            f"Current title: {current_document.title}\n"
            f"Current type: {current_document.document_type}\n"
            f"Current pages: {current_document.page_range.start_page}-"
            f"{current_document.page_range.end_page}"
        )
    return (
        (
            "You classify page boundaries in scanned insurance claim files. "
            "Decide whether the current page continues the current document "
            "or starts a new one."
        ),
        (
            "Classify the current page and use its number for page_number.\n\n"
            f"Prior page number: {prior_page.page_number}\n"
            f"Prior page text:\n{_clip(prior_page.text, 3000)}\n\n"
            f"Current page number: {page.page_number}\n"
            f"Current page text:\n{_clip(page.text, 3000)}\n\n"
            f"{current_context}"
        ),
    )


def build_extract_metadata_prompt(
    document: LogicalDocument,
    chunks: list[DocumentChunk],
) -> tuple[str, str]:
    chunk_text = "\n\n".join(
        f"Source ref: {chunk.source_ref}\n{chunk.text}" for chunk in chunks
    )
    return (
        (
            "You extract concise metadata for logical documents in scanned "
            "insurance claim files."
        ),
        (
            "Extract a title, summary of no more than 200 words, involved "
            "parties, useful events, and document type. Every event must use "
            "a provided chunk source_ref. Keep the initial document type when "
            "it is not unknown.\n\n"
            f"Document id: {document.id}\n"
            f"Page range: {document.page_range.start_page}-"
            f"{document.page_range.end_page}\n"
            f"Initial title: {document.title}\n"
            f"Initial document_type: {document.document_type}\n\n"
            f"Chunks:\n{_clip(chunk_text, 10000)}"
        ),
    )


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"
