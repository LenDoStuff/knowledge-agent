"""OCR text chunking with page references."""

from __future__ import annotations

from claim_kb.schemas import (
    DocumentChunk,
    LogicalDocument,
    PageRange,
    page_id_for,
    source_ref_for,
)


def chunk_documents(
    claim_id: str,
    documents: list[LogicalDocument],
    max_chars: int = 1600,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            _chunk_document(claim_id, document, len(chunks), max_chars)
        )
    return chunks


def _chunk_document(
    claim_id: str,
    document: LogicalDocument,
    global_start_index: int,
    max_chars: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_page_ids: list[str] = []
    start_page: int | None = None
    end_page: int | None = None

    def flush() -> None:
        nonlocal current_parts, current_page_ids, start_page, end_page
        if start_page is None or end_page is None:
            return
        text = "\n\n".join(part for part in current_parts if part).strip()
        if not text:
            return
        chunk_index = len(chunks)
        chunk_id = f"{document.id}-CHUNK-{chunk_index + 1:03d}"
        page_range = PageRange(start_page=start_page, end_page=end_page)
        chunks.append(
            DocumentChunk(
                claim_id=claim_id,
                document_id=document.id,
                chunk_id=chunk_id,
                source_ref=source_ref_for(claim_id, document.id, chunk_id),
                chunk_index=global_start_index + chunk_index,
                document_type=document.document_type,
                page_range=page_range,
                page_ids=current_page_ids,
                text=text,
            )
        )
        current_parts = []
        current_page_ids = []
        start_page = None
        end_page = None

    for page in document.pages:
        page_text = page.text.strip()
        if not page_text:
            continue
        candidate_length = sum(len(part) for part in current_parts) + len(page_text)
        if current_parts and candidate_length > max_chars:
            flush()
        if start_page is None:
            start_page = page.page_number
        end_page = page.page_number
        current_page_ids.append(page_id_for(claim_id, page.page_number))
        current_parts.append(f"Page {page.page_number}\n{page_text}")
    flush()

    return chunks
