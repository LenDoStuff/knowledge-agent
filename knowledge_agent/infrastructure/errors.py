"""Normalized application errors for LLM provider failures."""

from __future__ import annotations


class LlmError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.request_id = request_id
        self.status_code = status_code


class LlmAuthenticationError(LlmError):
    """The provider rejected application credentials."""


class LlmRateLimitError(LlmError):
    """The provider rejected the request due to a rate limit."""


class LlmTimeoutError(LlmError):
    """The provider request timed out."""


class LlmConnectionError(LlmError):
    """The provider could not be reached."""


class LlmUnsupportedRequestError(LlmError):
    """The provider rejected the model or request parameters."""


class LlmIncompleteResponseError(LlmError):
    """The response ended before completion."""


class LlmStructuredOutputError(LlmError):
    """The response did not validate against the requested schema."""


class LlmRefusalError(LlmError):
    """The provider returned a model refusal."""


class LlmProviderError(LlmError):
    """The provider returned an otherwise unclassified failure."""
