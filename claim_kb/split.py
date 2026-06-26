"""PDF splitting and logical document grouping."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter

from claim_kb.classify import ClaimClassifier
from claim_kb.schemas import LogicalDocument, OcrPage, PageRange
from claim_kb.storage import slugify


def group_logical_documents(
    claim_id: str,
    pages: list[OcrPage],
    classifier: ClaimClassifier,
) -> list[LogicalDocument]:
    sorted_pages = sorted(pages, key=lambda item: item.page_number)
    documents: list[LogicalDocument] = []
    current: LogicalDocument | None = None
    prior_page: OcrPage | None = None

    for page in sorted_pages:
        decision = classifier.classify_page_boundary(page, prior_page, current)
        is_new_document = current is None or decision.is_new_document
        if is_new_document:
            document_id = f"DOC-{len(documents) + 1:03d}"
            current = LogicalDocument(
                id=document_id,
                title=decision.title,
                document_type=decision.document_type,
                page_range=PageRange(
                    start_page=page.page_number,
                    end_page=page.page_number,
                ),
                pages=[page],
            )
            documents.append(current)
        else:
            assert current is not None
            current.pages.append(page)
            current.page_range = PageRange(
                start_page=current.page_range.start_page,
                end_page=page.page_number,
            )
            if current.document_type == "unknown" and decision.document_type != "unknown":
                current.document_type = decision.document_type
            if current.title in {"unknown", "Claim document", "Untitled document"}:
                current.title = decision.title
        prior_page = page

    for document in documents:
        document.page_range = PageRange(
            start_page=min(page.page_number for page in document.pages),
            end_page=max(page.page_number for page in document.pages),
        )
    return documents


def write_split_pdfs(
    original_pdf_path: Path,
    documents: list[LogicalDocument],
    output_dir: Path,
) -> list[LogicalDocument]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(original_pdf_path))
    page_count = len(reader.pages)
    for document in documents:
        writer = PdfWriter()
        for page_number in range(
            document.page_range.start_page, document.page_range.end_page + 1
        ):
            if page_number < 1 or page_number > page_count:
                raise ValueError(
                    f"Page {page_number} is outside PDF page count {page_count}"
                )
            writer.add_page(reader.pages[page_number - 1])
        file_name = f"{document.id}_{slugify(document.document_type)}.pdf"
        output_path = output_dir / file_name
        with output_path.open("wb") as handle:
            writer.write(handle)
        document.file_name = file_name
    return documents
