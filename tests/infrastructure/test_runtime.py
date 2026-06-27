from types import SimpleNamespace

from infrastructure.config import LlmSettings
from infrastructure.runtime import (
    OPENROUTER_BASE_URL,
    create_browser_credential,
    create_openai_runtime,
)


def openrouter_settings() -> LlmSettings:
    return LlmSettings(
        mode="home",
        model="provider/model",
        reasoning_effort="medium",
        openrouter_api_key="secret-test-key",
        azure_ai_project_endpoint=None,
    )


def azure_settings() -> LlmSettings:
    return LlmSettings(
        mode="work",
        model="deployment-name",
        reasoning_effort="medium",
        openrouter_api_key=None,
        azure_ai_project_endpoint=(
            "https://example.services.ai.azure.com/api/projects/proj"
        ),
    )


class FakeResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_browser_credential_factory_is_explicit(monkeypatch):
    credential = FakeResource()
    monkeypatch.setattr(
        "infrastructure.runtime.InteractiveBrowserCredential",
        lambda: credential,
    )

    assert create_browser_credential() is credential


def test_factory_builds_openrouter_client_and_closes_it(monkeypatch):
    calls = []
    client = FakeResource()

    def build_client(**kwargs):
        calls.append(kwargs)
        return client

    monkeypatch.setattr("infrastructure.runtime.OpenAI", build_client)

    runtime = create_openai_runtime(openrouter_settings())
    runtime.close()

    assert runtime.client is client
    assert calls == [
        {
            "api_key": "secret-test-key",
            "base_url": OPENROUTER_BASE_URL,
            "max_retries": 0,
        }
    ]
    assert client.closed


def test_factory_builds_azure_client_with_browser_credential(monkeypatch):
    credential = FakeResource()
    project = FakeResource()
    client = FakeResource()
    configured_client = FakeResource()
    calls = []

    project.get_openai_client = lambda: SimpleNamespace(
        with_options=lambda **kwargs: calls.append(("options", kwargs))
        or configured_client
    )

    def build_project(endpoint, credential):
        calls.append(("project", endpoint, credential))
        return project

    monkeypatch.setattr(
        "infrastructure.runtime.create_browser_credential",
        lambda: credential,
    )
    monkeypatch.setattr(
        "infrastructure.runtime.AIProjectClient",
        build_project,
    )

    runtime = create_openai_runtime(azure_settings())
    assert runtime.project_client is project
    assert runtime.azure_credential is credential
    runtime.close()

    assert runtime.client is configured_client
    assert calls == [
        ("project", azure_settings().azure_ai_project_endpoint, credential),
        ("options", {"max_retries": 0}),
    ]
    assert configured_client.closed
    assert project.closed
    assert credential.closed
