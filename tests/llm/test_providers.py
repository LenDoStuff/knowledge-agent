import asyncio

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import (
    create_browser_credential,
    open_agent_runtime,
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


def test_runtime_builds_nvidia_chat_model_and_closes_client():
    with open_agent_runtime(nvidia_settings()) as runtime:
        client = runtime.openai
        assert isinstance(runtime.model, OpenAIChatModel)
        assert str(client.base_url) == "https://integrate.api.nvidia.com/v1/"
        assert client.max_retries == 0
        assert runtime.model.settings["temperature"] == 0
        assert runtime.model.settings["thinking"] is False
        assert runtime.azure_project is None
        assert runtime.azure_credential is None
        assert not client.is_closed()
    assert client.is_closed()


def test_runtime_can_open_and_close_inside_a_running_event_loop():
    async def use_runtime():
        with open_agent_runtime(nvidia_settings()) as runtime:
            assert runtime.thread.is_alive()

    asyncio.run(use_runtime())


def test_runtime_builds_azure_responses_model_with_project_resources(monkeypatch):
    credential = FakeResource()
    project = FakeResource()
    calls = []

    def build_project(endpoint, credential):
        calls.append((endpoint, credential))
        return project

    monkeypatch.setattr(
        "knowledge_agent.llm.providers.create_browser_credential",
        lambda: credential,
    )
    monkeypatch.setattr(
        "knowledge_agent.llm.providers.AIProjectClient",
        build_project,
    )

    with open_agent_runtime(azure_settings()) as runtime:
        client = runtime.openai
        assert isinstance(runtime.model, OpenAIResponsesModel)
        assert runtime.azure_project is project
        assert runtime.azure_credential is credential
        assert str(client.base_url).endswith("/api/projects/proj/openai/v1/")
        assert client.max_retries == 0
        assert runtime.model.settings["openai_reasoning_effort"] == "medium"

    assert calls == [(azure_settings().azure_ai_project_endpoint, credential)]
    assert client.is_closed()
    assert project.closed
    assert credential.closed
