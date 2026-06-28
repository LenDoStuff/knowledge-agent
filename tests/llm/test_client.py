import logging
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from knowledge_agent.llm.client import LlmError, ResponsesClient
from knowledge_agent.llm.config import LlmSettings


class Answer(BaseModel):
    city: str
    country: str


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_client(result):
    responses = FakeResponses(result)
    return SimpleNamespace(responses=responses), responses


def settings() -> LlmSettings:
    return LlmSettings(
        profile="api_key",
        model="provider/model",
        reasoning_effort="medium",
        openrouter_api_key="secret-test-key",
    )


def completed_response(parsed=None):
    return SimpleNamespace(
        output_parsed=parsed,
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
    raw_client, responses = fake_client(
        completed_response(Answer(city="Paris", country="France"))
    )
    client = ResponsesClient(settings(), raw_client)
    with caplog.at_level(logging.INFO):
        result = client.parse("system", "user", Answer)

    request = responses.calls[0]
    assert result == Answer(city="Paris", country="France")
    assert request["model"] == "provider/model"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["text_format"] is Answer
    assert "provider=openrouter" in caplog.text
    assert "request_id=req_test" in caplog.text
    assert "reasoning_tokens=4" in caplog.text
    assert "secret-test-key" not in caplog.text


def _status_error(error_type, status_code):
    request = httpx.Request("POST", "https://provider.example/v1/responses")
    response = httpx.Response(status_code, request=request)
    return error_type("secret-test-key", response=response, body=None)


@pytest.mark.parametrize(
    ("provider_error", "category"),
    [
        (_status_error(openai.AuthenticationError, 401), "authentication"),
        (_status_error(openai.RateLimitError, 429), "rate_limit"),
        (_status_error(openai.BadRequestError, 400), "unsupported_request"),
        (
            openai.APITimeoutError(
                httpx.Request("POST", "https://provider.example/v1/responses")
            ),
            "timeout",
        ),
        (
            openai.APIConnectionError(
                request=httpx.Request(
                    "POST", "https://provider.example/v1/responses"
                )
            ),
            "connection",
        ),
    ],
)
def test_provider_errors_are_normalized_without_secret_details(
    caplog, provider_error, category
):
    raw_client, _ = fake_client(provider_error)
    client = ResponsesClient(settings(), raw_client)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(LlmError) as raised:
            client.parse("system", "user", Answer)
    assert raised.value.category == category
    assert "secret-test-key" not in str(raised.value)
    assert "secret-test-key" not in caplog.text


def test_structured_validation_error_is_normalized():
    with pytest.raises(ValidationError) as validation:
        Answer.model_validate({"city": "Paris"})
    raw_client, _ = fake_client(validation.value)
    with pytest.raises(LlmError) as raised:
        ResponsesClient(settings(), raw_client).parse("system", "user", Answer)
    assert raised.value.category == "output"


def test_incomplete_and_refusal_responses_are_explicit():
    incomplete = completed_response()
    incomplete.status = "incomplete"
    incomplete.incomplete_details = SimpleNamespace(reason="provider_limit")
    raw_client, _ = fake_client(incomplete)
    with pytest.raises(LlmError) as raised:
        ResponsesClient(settings(), raw_client).parse("system", "user", Answer)
    assert raised.value.category == "incomplete"

    refusal = completed_response()
    refusal.output = [
        SimpleNamespace(
            content=[SimpleNamespace(type="refusal", refusal="Cannot comply")]
        )
    ]
    raw_client, _ = fake_client(refusal)
    with pytest.raises(LlmError) as raised:
        ResponsesClient(settings(), raw_client).parse("system", "user", Answer)
    assert raised.value.category == "refusal"
