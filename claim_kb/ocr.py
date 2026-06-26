"""Azure Document Intelligence OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from claim_kb.config import ClaimKbSettings
from claim_kb.schemas import OcrPage


class OcrClient(Protocol):
    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[OcrPage]:
        ...


class AzureDocumentIntelligenceOcrClient:
    def __init__(self, settings: ClaimKbSettings, credential: object) -> None:
        settings.validate_document_intelligence_endpoint()
        from azure.ai.documentintelligence import DocumentIntelligenceClient

        self._client = DocumentIntelligenceClient(
            endpoint=settings.document_intelligence_endpoint,
            credential=credential,
        )

    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[OcrPage]:
        with pdf_path.open("rb") as handle:
            poller = self._client.begin_analyze_document("prebuilt-layout", body=handle)
            result = poller.result()
        pages: list[OcrPage] = []
        for page in getattr(result, "pages", []) or []:
            lines = [line.content for line in getattr(page, "lines", []) or []]
            words = getattr(page, "words", []) or []
            pages.append(
                OcrPage(
                    claim_id=claim_id,
                    page_number=int(page.page_number),
                    text="\n".join(lines).strip(),
                    lines=lines,
                    width=getattr(page, "width", None),
                    height=getattr(page, "height", None),
                    unit=getattr(page, "unit", None),
                    word_count=len(words),
                )
            )
        return sorted(pages, key=lambda item: item.page_number)
