"""Azure Document Intelligence OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from claim_kb.config import ClaimKbSettings
from claim_kb.schemas import PageText, page_id_for


class OcrClient(Protocol):
    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        ...


class AzureDocumentIntelligenceOcrClient:
    def __init__(self, settings: ClaimKbSettings, credential: object) -> None:
        settings.validate_document_intelligence_endpoint()
        from azure.ai.documentintelligence import DocumentIntelligenceClient

        self._client = DocumentIntelligenceClient(
            endpoint=settings.document_intelligence_endpoint,
            credential=credential,
        )

    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        with pdf_path.open("rb") as handle:
            poller = self._client.begin_analyze_document("prebuilt-layout", body=handle)
            result = poller.result()
        pages: list[PageText] = []
        for page in result.pages:
            text = "\n".join(line.content for line in page.lines).strip()
            pages.append(
                PageText(
                    claim_id=claim_id,
                    page_number=int(page.page_number),
                    page_id=page_id_for(claim_id, int(page.page_number)),
                    text=text,
                    width=page.width,
                    height=page.height,
                    unit=page.unit,
                    word_count=len(page.words),
                )
            )
        return sorted(pages, key=lambda item: item.page_number)
