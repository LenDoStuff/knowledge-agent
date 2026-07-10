import pytest

from knowledge_agent.config import ConfigurationError, load_profile
from knowledge_agent.llm.config import (
    NVIDIA_DEEPSEEK_MODEL,
    llm_provider,
    load_llm_settings,
)


ENV_NAMES = [
    "KNOWLEDGE_AGENT_PROFILE",
    "nvidia_base_url",
    "nvidia_api_key_ds4",
    "AZURE_AI_PROJECT_ENDPOINT",
    "AZURE_OPENAI_MODEL",
    "LLM_REASONING_EFFORT",
]


def set_environment(monkeypatch, values):
    monkeypatch.setattr("knowledge_agent.config.load_dotenv", lambda: None)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_api_key_settings_load_without_exposing_secret(monkeypatch):
    set_environment(
        monkeypatch,
        {
            "KNOWLEDGE_AGENT_PROFILE": "api_key",
            "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
            "nvidia_api_key_ds4": "secret-test-key",
            "LLM_REASONING_EFFORT": "high",
        },
    )
    profile = load_profile()
    settings = load_llm_settings(profile)
    assert settings.profile == "api_key"
    assert llm_provider(settings) == "nvidia"
    assert settings.model == NVIDIA_DEEPSEEK_MODEL
    assert settings.reasoning_effort == "high"
    assert "secret-test-key" not in repr(settings)


def test_azure_project_settings_select_azure_model(monkeypatch):
    set_environment(
        monkeypatch,
        {
            "KNOWLEDGE_AGENT_PROFILE": "azure_project",
            "AZURE_OPENAI_MODEL": "deployment-name",
            "AZURE_AI_PROJECT_ENDPOINT": "https://project.example",
        },
    )
    settings = load_llm_settings(load_profile())
    assert settings.profile == "azure_project"
    assert llm_provider(settings) == "azure"
    assert settings.model == "deployment-name"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "KNOWLEDGE_AGENT_PROFILE"),
        ({"KNOWLEDGE_AGENT_PROFILE": "unknown"}, "KNOWLEDGE_AGENT_PROFILE"),
        (
            {
                "KNOWLEDGE_AGENT_PROFILE": "api_key",
                "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
            },
            "nvidia_api_key_ds4",
        ),
        (
            {
                "KNOWLEDGE_AGENT_PROFILE": "azure_project",
                "AZURE_OPENAI_MODEL": "deployment",
            },
            "AZURE_AI_PROJECT_ENDPOINT",
        ),
        (
            {
                "KNOWLEDGE_AGENT_PROFILE": "api_key",
                "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
                "nvidia_api_key_ds4": "key",
                "LLM_REASONING_EFFORT": "extreme",
            },
            "LLM_REASONING_EFFORT",
        ),
    ],
)
def test_invalid_configuration_fails_at_startup(monkeypatch, environment, message):
    set_environment(monkeypatch, environment)
    with pytest.raises(ConfigurationError, match=message):
        profile = load_profile()
        load_llm_settings(profile)
