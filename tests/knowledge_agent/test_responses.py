import logging
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from knowledge_agent.infrastructure.config import LlmSettings
from knowledge_agent.infrastructure.errors import (
    LlmAuthenticationError,
    LlmConnectionError,
    LlmIncompleteResponseError,
    LlmRateLimitError,
    LlmRefusalError,
    LlmStructuredOutputError,
    LlmTimeoutError,
    LlmUnsupportedRequestError,
)
from knowledge_agent.infrastructure.responses import PortableResponsesClient


class Answer(BaseModel):
    city: str
    country: str


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeRuntime:
    def __init__(self, result):
        self.responses = FakeResponses(result)
        self.client = SimpleNamespace(responses=self.responses)
        self.closed = False

    def close(self):
        self.closed = True


def settings() -> LlmSettings:
    return LlmSettings(
        mode="home",
        model="provider/model",
        reasoning_effort="medium",
        openrouter_api_key="secret-test-key",
        azure_ai_project_endpoint=None,
    )


def completed_response(parsed=None, output_text="Paris, France"):
    return SimpleNamespace(
        output_parsed=parsed,
        output_text=output_text,
        output=[],
        status="completed",
        incomplete_details=None,
        _request_id="req_test",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=4),
        ),
    )


def test_parse_returns_pydantic_type(caplog):
    runtime = FakeRuntime(completed_response(Answer(city="Paris", country="France")))
    client = PortableResponsesClient(settings(), runtime)

    with caplog.at_level(logging.INFO):
        result = client.parse("system", "user", Answer)

    method, request = runtime.responses.calls[0]
    assert method == "parse"
    assert result == Answer(city="Paris", country="France")
    assert request["model"] == "provider/model"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["text_format"] is Answer
    assert "provider=openrouter" in caplog.text
    assert "request_id=req_test" in caplog.text
    assert "status=completed" in caplog.text
    assert "reasoning_tokens=4" in caplog.text
    assert "latency_ms=" in caplog.text
    assert "retry_count=0" in caplog.text
    assert "secret-test-key" not in caplog.text


def test_create_returns_text():
    runtime = FakeRuntime(completed_response())
    client = PortableResponsesClient(settings(), runtime)

    result = client.create("Give the capital of France.")

    assert result == "Paris, France"


def _status_error(error_type, status_code):
    request = httpx.Request("POST", "https://provider.example/v1/responses")
    response = httpx.Response(
        status_code,
        request=request,
        headers={"x-request-id": "req_error"},
    )
    return error_type(
        "provider detail containing secret-test-key",
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    ("provider_error", "application_error"),
    [
        (_status_error(openai.AuthenticationError, 401), LlmAuthenticationError),
        (_status_error(openai.RateLimitError, 429), LlmRateLimitError),
        (_status_error(openai.BadRequestError, 400), LlmUnsupportedRequestError),
        (_status_error(openai.NotFoundError, 404), LlmUnsupportedRequestError),
        (
            openai.APITimeoutError(
                httpx.Request("POST", "https://provider.example/v1/responses")
            ),
            LlmTimeoutError,
        ),
        (
            openai.APIConnectionError(
                request=httpx.Request(
                    "POST",
                    "https://provider.example/v1/responses",
                )
            ),
            LlmConnectionError,
        ),
    ],
)
def test_provider_errors_are_normalized_without_secret_details(
    caplog,
    provider_error,
    application_error,
):
    client = PortableResponsesClient(settings(), FakeRuntime(provider_error))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(application_error) as raised:
            client.parse("system", "user", Answer)

    assert "secret-test-key" not in str(raised.value)
    assert "secret-test-key" not in caplog.text


def test_structured_validation_error_is_normalized():
    try:
        Answer.model_validate({"city": "Paris"})
    except ValidationError as validation_error:
        error = validation_error
    client = PortableResponsesClient(settings(), FakeRuntime(error))

    with pytest.raises(LlmStructuredOutputError):
        client.parse("system", "user", Answer)


def test_incomplete_response_reports_output_limit():
    response = completed_response()
    response.status = "incomplete"
    response.incomplete_details = SimpleNamespace(reason="provider_limit")
    client = PortableResponsesClient(settings(), FakeRuntime(response))

    with pytest.raises(LlmIncompleteResponseError, match="provider_limit"):
        client.parse("system", "user", Answer)


def test_refusal_is_normalized():
    response = completed_response(parsed=None)
    response.output = [
        SimpleNamespace(
            content=[SimpleNamespace(type="refusal", refusal="Cannot comply")]
        )
    ]
    client = PortableResponsesClient(settings(), FakeRuntime(response))

    with pytest.raises(LlmRefusalError):
        client.parse("system", "user", Answer)


def test_client_closes_runtime():
    runtime = FakeRuntime(completed_response())
    client = PortableResponsesClient(settings(), runtime)

    client.close()

    assert runtime.closed
