import logging
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from knowledge_agent.llm.client import LlmError, parse_structured_output
from knowledge_agent.llm.config import LlmSettings


class Answer(BaseModel):
    city: str
    country: str


class FakeEndpoint:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

def fake_responses_client(result):
    endpoint = FakeEndpoint(result)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeEndpoint(AssertionError("Chat API should not be used"))
        ),
        responses=endpoint,
    )
    return client, endpoint


def nvidia_settings() -> LlmSettings:
    return LlmSettings(
        profile="api_key",
        model="deepseek-ai/deepseek-v4-pro",
        reasoning_effort="low",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_api_key_ds4="secret-test-key",
    )


def azure_settings() -> LlmSettings:
    return LlmSettings(
        profile="azure_project",
        model="deployment-name",
        reasoning_effort="medium",
        azure_ai_project_endpoint="https://example.services.ai.azure.com/api/projects/p",
    )


def completed_response(parsed=None):
    return SimpleNamespace(
        output_parsed=parsed,
        output=[],
        status="completed",
        incomplete_details=None,
        _request_id="req_responses",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=4),
        ),
    )


def test_nvidia_uses_responses_structured_output(caplog):
    raw_client, endpoint = fake_responses_client(
        completed_response(Answer(city="Paris", country="France"))
    )
    with caplog.at_level(logging.INFO):
        result = parse_structured_output(nvidia_settings(), raw_client, "system", "user", Answer)

    request = endpoint.calls[0]
    assert result == Answer(city="Paris", country="France")
    assert request["model"] == "deepseek-ai/deepseek-v4-pro"
    assert request["reasoning"] == {"effort": "low"}
    assert request["text_format"] is Answer
    assert request["input"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert "provider=nvidia" in caplog.text
    assert "request_id=req_responses" in caplog.text
    assert "secret-test-key" not in caplog.text


def test_azure_uses_responses_structured_output():
    raw_client, endpoint = fake_responses_client(
        completed_response(Answer(city="Paris", country="France"))
    )

    result = parse_structured_output(
        azure_settings(), raw_client, "system", "user", Answer
    )

    assert result == Answer(city="Paris", country="France")
    assert endpoint.calls[0]["model"] == "deployment-name"
    assert endpoint.calls[0]["reasoning"] == {"effort": "medium"}
    assert endpoint.calls[0]["text_format"] is Answer


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
                httpx.Request("POST", "https://provider.example/v1/chat/completions")
            ),
            "timeout",
        ),
        (
            openai.APIConnectionError(
                request=httpx.Request(
                    "POST", "https://provider.example/v1/chat/completions"
                )
            ),
            "connection",
        ),
    ],
)
def test_provider_errors_are_normalized_without_secret_details(
    caplog, provider_error, category
):
    raw_client, _ = fake_responses_client(provider_error)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(LlmError) as raised:
            parse_structured_output(
                nvidia_settings(), raw_client, "system", "user", Answer
            )
    assert raised.value.category == category
    assert "secret-test-key" not in str(raised.value)
    assert "secret-test-key" not in caplog.text


def test_structured_validation_error_is_normalized():
    with pytest.raises(ValidationError) as validation:
        Answer.model_validate({"city": "Paris"})
    raw_client, _ = fake_responses_client(validation.value)
    with pytest.raises(LlmError) as raised:
        parse_structured_output(nvidia_settings(), raw_client, "system", "user", Answer)
    assert raised.value.category == "output"


@pytest.mark.parametrize(
    "settings",
    [nvidia_settings(), azure_settings()],
    ids=["nvidia", "azure"],
)
def test_responses_incomplete_and_refusal_responses_are_explicit(settings):
    incomplete = completed_response()
    incomplete.status = "incomplete"
    incomplete.incomplete_details = SimpleNamespace(reason="provider_limit")
    raw_client, _ = fake_responses_client(incomplete)
    with pytest.raises(LlmError) as raised:
        parse_structured_output(settings, raw_client, "system", "user", Answer)
    assert raised.value.category == "incomplete"

    refusal = completed_response()
    refusal.output = [
        SimpleNamespace(
            content=[SimpleNamespace(type="refusal", refusal="Cannot comply")]
        )
    ]
    raw_client, _ = fake_responses_client(refusal)
    with pytest.raises(LlmError) as raised:
        parse_structured_output(settings, raw_client, "system", "user", Answer)
    assert raised.value.category == "refusal"
