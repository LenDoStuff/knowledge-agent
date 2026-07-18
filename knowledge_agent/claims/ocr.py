"""Azure and Snowflake OCR adapters normalized to persisted claim pages."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from knowledge_agent.claims.config import validate_snowflake_stage_name
from knowledge_agent.claims.models import PageText, page_id_for


LOGGER = logging.getLogger(__name__)
SNOWFLAKE_PARSE_DOCUMENT_SQL = """
SELECT AI_PARSE_DOCUMENT(
    TO_FILE(?, ?),
    OBJECT_CONSTRUCT('mode', 'OCR', 'page_split', TRUE),
    TRUE
) AS PARSED_DOCUMENT
"""


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
        LOGGER.info("ocr_start claim_id=%s document=%s", claim_id, pdf_path.name)
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
                )
            )
        pages = sorted(pages, key=lambda item: item.page_number)
        LOGGER.info(
            "ocr_complete claim_id=%s document=%s pages=%d",
            claim_id,
            pdf_path.name,
            len(pages),
        )
        return pages

    def close(self) -> None:
        self._client.close()


class SnowflakeParseDocumentOcrClient:
    """Parse PDFs through an application-managed Snowflake internal stage."""

    def __init__(self, session, stage_name: str) -> None:
        validate_snowflake_stage_name(stage_name)
        if not session.get_current_database() or not session.get_current_schema():
            raise ValueError(
                "Snowflake OCR requires database and schema in the named connection"
            )
        self._session = session
        self._stage_name = stage_name
        self._session.sql(
            "CREATE STAGE IF NOT EXISTS IDENTIFIER(?)",
            params=[stage_name],
        ).collect()

    def extract_pages(self, claim_id: str, pdf_path: Path) -> list[PageText]:
        LOGGER.info("ocr_start claim_id=%s document=%s", claim_id, pdf_path.name)
        upload_prefix = f"knowledge-agent/{uuid4().hex}"
        stage_location = f"@{self._stage_name}/{upload_prefix}"
        relative_path = f"{upload_prefix}/{pdf_path.name}"
        try:
            results = self._session.file.put(
                str(pdf_path),
                stage_location,
                auto_compress=False,
                overwrite=False,
            )
            if len(results) != 1:
                raise RuntimeError(
                    "Snowflake OCR expected exactly one staged PDF upload result"
                )
            status = str(getattr(results[0], "status", "")).upper()
            if status != "UPLOADED":
                message = str(getattr(results[0], "message", "")).strip()
                raise RuntimeError(
                    f"Snowflake OCR PDF upload did not complete: {status or 'UNKNOWN'}"
                    + (f" ({message})" if message else "")
                )

            rows = self._session.sql(
                SNOWFLAKE_PARSE_DOCUMENT_SQL,
                params=[f"@{self._stage_name}", relative_path],
            ).collect()
            pages = _snowflake_pages(claim_id, rows)
        finally:
            self._session.file.remove(stage_location)

        LOGGER.info(
            "ocr_complete claim_id=%s document=%s pages=%d",
            claim_id,
            pdf_path.name,
            len(pages),
        )
        return pages

    def close(self) -> None:
        """Leave session shutdown to the owning AgentRuntime."""


def _snowflake_pages(claim_id: str, rows: list[Any]) -> list[PageText]:
    if len(rows) != 1:
        raise ValueError("Snowflake OCR must return exactly one result row")
    raw_result = _row_value(rows[0], "PARSED_DOCUMENT")
    result = _json_object(raw_result, "Snowflake OCR result")
    error = result.get("error")
    if error is not None:
        raise RuntimeError(f"Snowflake OCR failed: {error}")

    value = _json_object(result.get("value"), "Snowflake OCR value")
    raw_pages = value.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("Snowflake OCR value must contain a non-empty pages list")
    metadata = _json_object(result.get("metadata"), "Snowflake OCR metadata")
    page_count = metadata.get("pageCount")
    if isinstance(page_count, bool) or not isinstance(page_count, int):
        raise ValueError("Snowflake OCR metadata.pageCount must be an integer")
    if page_count != len(raw_pages):
        raise ValueError("Snowflake OCR page count does not match returned pages")

    pages_by_index: dict[int, str] = {}
    for raw_page in raw_pages:
        page = _json_object(raw_page, "Snowflake OCR page")
        index = page.get("index")
        content = page.get("content")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Snowflake OCR page index must be an integer")
        if not isinstance(content, str):
            raise ValueError("Snowflake OCR page content must be text")
        if index in pages_by_index:
            raise ValueError(f"Snowflake OCR returned duplicate page index {index}")
        pages_by_index[index] = content.strip()

    expected_indexes = list(range(page_count))
    if sorted(pages_by_index) != expected_indexes:
        raise ValueError("Snowflake OCR page indexes must be contiguous from zero")
    return [
        PageText(
            claim_id=claim_id,
            page_number=index + 1,
            page_id=page_id_for(claim_id, index + 1),
            text=pages_by_index[index],
        )
        for index in expected_indexes
    ]


def _row_value(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError):
        try:
            return getattr(row, name)
        except AttributeError as exc:
            raise ValueError(f"Snowflake OCR row is missing {name}") from exc


def _json_object(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value
