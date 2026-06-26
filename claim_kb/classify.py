"""LLM-assisted document boundary and metadata extraction."""

from __future__ import annotations

import json
from typing import Any, Protocol

from claim_kb.config import ClaimKbSettings
from claim_kb.schemas import (
    DocumentMetadata,
    LogicalDocument,
    OcrPage,
    PageBoundaryDecision,
)


class ClaimClassifier(Protocol):
    def classify_page_boundary(
        self,
        page: OcrPage,
        prior_page: OcrPage | None,
        current_document: LogicalDocument | None,
    ) -> PageBoundaryDecision:
        ...

    def extract_document_metadata(self, document: LogicalDocument) -> DocumentMetadata:
        ...


class AzureClaimClassifier:
    def __init__(self, settings: ClaimKbSettings, credential: object) -> None:
        if not settings.ai_project_endpoint or not settings.chat_deployment:
            raise ValueError("AI project endpoint and chat deployment are required")
        from azure.ai.projects import AIProjectClient

        project = AIProjectClient(
            endpoint=settings.ai_project_endpoint,
            credential=credential,
        )
        self._client = project.get_openai_client()
        self._model = settings.chat_deployment

    def classify_page_boundary(
        self,
        page: OcrPage,
        prior_page: OcrPage | None,
        current_document: LogicalDocument | None,
    ) -> PageBoundaryDecision:
        if prior_page is None:
            return PageBoundaryDecision(
                page_number=page.page_number,
                is_new_document=True,
                document_type="unknown",
                title="Claim document",
                reason="First page in claim file",
                confidence=1.0,
            )

        current_context = "No current document."
        if current_document is not None:
            current_context = (
                f"Current document id: {current_document.id}\n"
                f"Current title: {current_document.title}\n"
                f"Current type: {current_document.document_type}\n"
                f"Current pages: {current_document.page_range.start_page}-"
                f"{current_document.page_range.end_page}"
            )

        payload = {
            "prior_page_number": prior_page.page_number,
            "prior_page_text": _clip(prior_page.text, 3000),
            "current_page_number": page.page_number,
            "current_page_text": _clip(page.text, 3000),
            "current_document_context": current_context,
        }
        response = self._json_chat(
            system=(
                "You classify page boundaries in scanned insurance claim files. "
                "Use the prior page and current document context to decide whether "
                "the current page continues the same logical document or starts a "
                "new one. Return strict JSON only."
            ),
            user=(
                "Return JSON with keys: is_new_document boolean, document_type "
                "string, title string, reason string, confidence number 0-1.\n\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        )
        response["page_number"] = page.page_number
        return PageBoundaryDecision.model_validate(response)

    def extract_document_metadata(self, document: LogicalDocument) -> DocumentMetadata:
        text = "\n\n".join(
            f"Page {page.page_number}\n{page.text}" for page in document.pages
        )
        response = self._json_chat(
            system=(
                "You extract concise metadata for logical documents in scanned "
                "insurance claim files. Return strict JSON only."
            ),
            user=(
                "Return JSON with keys: title string, summary string, "
                "involved_parties array of strings, document_type string.\n\n"
                f"Document id: {document.id}\n"
                f"Page range: {document.page_range.start_page}-"
                f"{document.page_range.end_page}\n"
                f"Initial title: {document.title}\n"
                f"Initial document_type: {document.document_type}\n\n"
                f"OCR text:\n{_clip(text, 10000)}"
            ),
        )
        if document.file_name is None:
            raise ValueError(f"Document {document.id} has no split PDF file name")
        return DocumentMetadata(
            id=document.id,
            title=_required_text(response, "title"),
            summary=_required_text(response, "summary", allow_empty=True),
            involved_parties=_required_text_list(response, "involved_parties"),
            document_type=_required_text(response, "document_type"),
            page_range=document.page_range,
            file_name=document.file_name,
        )

    def _json_chat(self, system: str, user: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return _parse_json_object(content)


def _parse_json_object(content: str | None) -> dict[str, Any]:
    if not content:
        raise ValueError("Expected JSON object content from chat completion")
    content = content.strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Expected chat completion to return a JSON object")
    return parsed


def _required_text(
    data: dict[str, Any],
    key: str,
    allow_empty: bool = False,
) -> str:
    if key not in data:
        raise ValueError(f"Missing required metadata field: {key}")
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"Metadata field {key} must be a string")
    value = value.strip()
    if not allow_empty and not value:
        raise ValueError(f"Metadata field {key} cannot be empty")
    return value


def _required_text_list(data: dict[str, Any], key: str) -> list[str]:
    if key not in data:
        raise ValueError(f"Missing required metadata field: {key}")
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"Metadata field {key} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Metadata field {key} must contain only strings")
    stripped = [item.strip() for item in value]
    if any(not item for item in stripped):
        raise ValueError(f"Metadata field {key} cannot contain empty strings")
    return stripped


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"
