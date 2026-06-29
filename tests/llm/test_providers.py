from types import SimpleNamespace

from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import (
    create_browser_credential,
    open_provider_clients,
)


def nvidia_settings() -> LlmSettings:
    return LlmSettings(
        profile="api_key",
        model="deepseek-ai/deepseek-v4-pro",
        reasoning_effort="medium",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_api_key_ds4="secret-test-key",
    )


def azure_settings() -> LlmSettings:
    return LlmSettings(
        profile="azure_project",
        model="deployment-name",
        reasoning_effort="medium",
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
        "knowledge_agent.llm.providers.InteractiveBrowserCredential",
        lambda: credential,
    )

    assert create_browser_credential() is credential


def test_factory_builds_nvidia_client_and_closes_it(monkeypatch):
    calls = []
    client = FakeResource()

    def build_client(**kwargs):
        calls.append(kwargs)
        return client

    monkeypatch.setattr("knowledge_agent.llm.providers.OpenAI", build_client)

    with open_provider_clients(nvidia_settings()) as provider:
        assert provider.openai is client
    assert calls == [
        {
            "api_key": "secret-test-key",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "max_retries": 0,
        }
    ]
    assert client.closed


def test_factory_builds_azure_client_with_browser_credential(monkeypatch):
    credential = FakeResource()
    project = FakeResource()
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
        "knowledge_agent.llm.providers.create_browser_credential",
        lambda: credential,
    )
    monkeypatch.setattr(
        "knowledge_agent.llm.providers.AIProjectClient",
        build_project,
    )

    with open_provider_clients(azure_settings()) as provider:
        assert provider.openai is configured_client
        assert provider.azure_project is project
        assert provider.azure_credential is credential

    assert calls == [
        ("project", azure_settings().azure_ai_project_endpoint, credential),
        ("options", {"max_retries": 0}),
    ]
    assert configured_client.closed
    assert project.closed
    assert credential.closed
