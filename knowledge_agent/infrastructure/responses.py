"""Portable OpenAI Responses API service."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from knowledge_agent.infrastructure.config import LlmSettings
from knowledge_agent.infrastructure.errors import (
    LlmAuthenticationError,
    LlmConnectionError,
    LlmError,
    LlmIncompleteResponseError,
    LlmProviderError,
    LlmRateLimitError,
    LlmRefusalError,
    LlmStructuredOutputError,
    LlmTimeoutError,
    LlmUnsupportedRequestError,
)
from knowledge_agent.infrastructure.runtime import OpenAiRuntime


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


class StructuredOutputClient(Protocol):
    def parse(
        self,
        system: str,
        user: str,
        response_model: type[ParsedModel],
    ) -> ParsedModel:
        ...


class PortableResponsesClient:
    def __init__(
        self,
        settings: LlmSettings,
        runtime: OpenAiRuntime,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._settings = settings
        self._runtime = runtime
        self._logger = logger

    def create(self, input: str | list[dict[str, str]]) -> str:
        response = self._request("create", input=input)
        refusal = _find_refusal(response)
        if refusal is not None:
            raise LlmRefusalError(
                "The model refused the request",
                provider=self._settings.provider,
                request_id=_request_id(response),
            )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise LlmProviderError(
                "The provider returned no text output",
                provider=self._settings.provider,
                request_id=_request_id(response),
            )
        return output_text

    def parse(
        self,
        system: str,
        user: str,
        response_model: type[ParsedModel],
    ) -> ParsedModel:
        response = self._request(
            "parse",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=response_model,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            if _find_refusal(response) is not None:
                raise LlmRefusalError(
                    "The model refused the structured request",
                    provider=self._settings.provider,
                    request_id=_request_id(response),
                )
            raise LlmStructuredOutputError(
                f"Missing structured output for {response_model.__name__}",
                provider=self._settings.provider,
                request_id=_request_id(response),
            )
        if not isinstance(parsed, response_model):
            raise LlmStructuredOutputError(
                f"Invalid structured output for {response_model.__name__}",
                provider=self._settings.provider,
                request_id=_request_id(response),
            )
        return parsed

    def close(self) -> None:
        self._runtime.close()

    def __enter__(self) -> "PortableResponsesClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _request(self, method: str, **kwargs: Any) -> Any:
        request = self._request_parameters(**kwargs)
        started = perf_counter()
        self._logger.info(
            "llm_request provider=%s model=%s retry_count=0",
            self._settings.provider,
            self._settings.model,
        )
        try:
            endpoint = getattr(self._runtime.client.responses, method)
            response = endpoint(**request)
        except Exception as exc:
            error = self._normalize_error(exc)
            self._logger.error(
                "llm_error provider=%s model=%s error_type=%s "
                "request_id=%s status_code=%s latency_ms=%d retry_count=0",
                self._settings.provider,
                self._settings.model,
                type(error).__name__,
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
            raise LlmIncompleteResponseError(
                f"The response was incomplete: {incomplete_reason or 'unknown'}",
                provider=self._settings.provider,
                request_id=request_id,
            )
        return response

    def _request_parameters(self, **kwargs: Any) -> dict[str, Any]:
        request = {
            "model": self._settings.model,
            "reasoning": {"effort": self._settings.reasoning_effort},
            **kwargs,
        }
        return request

    def _normalize_error(self, exc: Exception) -> LlmError:
        provider = self._settings.provider
        request_id = getattr(exc, "request_id", None)
        status_code = getattr(exc, "status_code", None)
        common = {
            "provider": provider,
            "request_id": request_id,
            "status_code": status_code,
        }
        if isinstance(
            exc,
            (openai.AuthenticationError, openai.PermissionDeniedError),
        ):
            return LlmAuthenticationError("LLM authentication failed", **common)
        if isinstance(exc, openai.RateLimitError):
            return LlmRateLimitError("LLM rate limit exceeded", **common)
        if isinstance(exc, openai.APITimeoutError):
            return LlmTimeoutError("LLM request timed out", **common)
        if isinstance(exc, openai.APIConnectionError):
            return LlmConnectionError("LLM connection failed", **common)
        if isinstance(
            exc,
            (
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
            ),
        ):
            return LlmUnsupportedRequestError(
                "The configured model or request parameters are unsupported",
                **common,
            )
        if isinstance(exc, (ValidationError, openai.APIResponseValidationError)):
            return LlmStructuredOutputError(
                "The provider returned invalid structured output",
                **common,
            )
        if isinstance(exc, openai.APIStatusError):
            return LlmProviderError("The LLM provider request failed", **common)
        if isinstance(exc, LlmError):
            return exc
        return LlmProviderError("The LLM request failed", **common)


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
