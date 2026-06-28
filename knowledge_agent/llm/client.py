"""Structured OpenAI Responses client."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import open_provider_clients


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


class LlmError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        category: str,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.category = category
        self.request_id = request_id
        self.status_code = status_code


class StructuredOutputClient(Protocol):
    def parse(
        self,
        system: str,
        user: str,
        response_model: type[ParsedModel],
    ) -> ParsedModel:
        ...


class ResponsesClient:
    def __init__(
        self,
        settings: LlmSettings,
        client: Any,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._settings = settings
        self._client = client
        self._logger = logger

    def parse(
        self,
        system: str,
        user: str,
        response_model: type[ParsedModel],
    ) -> ParsedModel:
        response = self._request(
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=response_model,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            category = "refusal" if _find_refusal(response) is not None else "output"
            raise LlmError(
                f"Missing structured output for {response_model.__name__}",
                provider=self._settings.provider,
                category=category,
                request_id=_request_id(response),
            )
        if not isinstance(parsed, response_model):
            raise LlmError(
                f"Invalid structured output for {response_model.__name__}",
                provider=self._settings.provider,
                category="output",
                request_id=_request_id(response),
            )
        return parsed

    def _request(self, **kwargs: Any) -> Any:
        request = {
            "model": self._settings.model,
            "reasoning": {"effort": self._settings.reasoning_effort},
            **kwargs,
        }
        started = perf_counter()
        self._logger.info(
            "llm_request provider=%s model=%s retry_count=0",
            self._settings.provider,
            self._settings.model,
        )
        try:
            response = self._client.responses.parse(**request)
        except Exception as exc:
            error = self._normalize_error(exc)
            self._logger.error(
                "llm_error provider=%s model=%s category=%s request_id=%s "
                "status_code=%s latency_ms=%d retry_count=0",
                self._settings.provider,
                self._settings.model,
                error.category,
                error.request_id,
                error.status_code,
                round((perf_counter() - started) * 1000),
            )
            raise error from None

        request_id = _request_id(response)
        status = getattr(response, "status", None)
        incomplete_reason = _incomplete_reason(response)
        usage = _usage(response)
        self._logger.info(
            "llm_response provider=%s model=%s request_id=%s status=%s "
            "incomplete_reason=%s input_tokens=%s output_tokens=%s "
            "reasoning_tokens=%s latency_ms=%d retry_count=0",
            self._settings.provider,
            self._settings.model,
            request_id,
            status,
            incomplete_reason,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["reasoning_tokens"],
            round((perf_counter() - started) * 1000),
        )
        if status == "incomplete":
            raise LlmError(
                f"The response was incomplete: {incomplete_reason or 'unknown'}",
                provider=self._settings.provider,
                category="incomplete",
                request_id=request_id,
            )
        return response

    def _normalize_error(self, exc: Exception) -> LlmError:
        if isinstance(exc, LlmError):
            return exc
        category = "provider"
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            category = "authentication"
        elif isinstance(exc, openai.RateLimitError):
            category = "rate_limit"
        elif isinstance(exc, openai.APITimeoutError):
            category = "timeout"
        elif isinstance(exc, openai.APIConnectionError):
            category = "connection"
        elif isinstance(
            exc,
            (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError),
        ):
            category = "unsupported_request"
        elif isinstance(exc, (ValidationError, openai.APIResponseValidationError)):
            category = "output"
        return LlmError(
            "The LLM request failed",
            provider=self._settings.provider,
            category=category,
            request_id=getattr(exc, "request_id", None),
            status_code=getattr(exc, "status_code", None),
        )


@contextmanager
def open_responses_client(settings: LlmSettings) -> Iterator[ResponsesClient]:
    with open_provider_clients(settings) as provider:
        yield ResponsesClient(settings, provider.openai)


def _request_id(response: Any) -> str | None:
    value = getattr(response, "_request_id", None)
    return str(value) if value else None


def _incomplete_reason(response: Any) -> str | None:
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    return str(reason) if reason else None


def _find_refusal(response: Any) -> str | None:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                refusal = getattr(content, "refusal", None)
                return str(refusal) if refusal else "refused"
    return None


def _usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
    }
