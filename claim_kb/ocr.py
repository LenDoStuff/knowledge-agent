"""Azure Document Intelligence OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from claim_kb.schemas import PageText, page_id_for


class OcrClient(Protocol):
    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        ...

    def close(self) -> None:
        ...


class AzureDocumentIntelligenceOcrClient:
    def __init__(self, endpoint: str, credential: object) -> None:
        from azure.ai.documentintelligence import DocumentIntelligenceClient

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint,
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

    def close(self) -> None:
        self._client.close()
