import pytest

from knowledge_agent.config import ConfigurationError, load_profile
from knowledge_agent.llm.config import LlmSettings


ENV_NAMES = [
    "KNOWLEDGE_AGENT_PROFILE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
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
            "OPENROUTER_MODEL": "provider/model",
            "OPENROUTER_API_KEY": "secret-test-key",
            "LLM_REASONING_EFFORT": "high",
        },
    )
    profile = load_profile()
    settings = LlmSettings.from_env(profile)
    assert settings.profile == "api_key"
    assert settings.provider == "openrouter"
    assert settings.model == "provider/model"
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
    settings = LlmSettings.from_env(load_profile())
    assert settings.profile == "azure_project"
    assert settings.provider == "azure"
    assert settings.model == "deployment-name"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "KNOWLEDGE_AGENT_PROFILE"),
        ({"KNOWLEDGE_AGENT_PROFILE": "unknown"}, "KNOWLEDGE_AGENT_PROFILE"),
        (
            {
                "KNOWLEDGE_AGENT_PROFILE": "api_key",
                "OPENROUTER_MODEL": "provider/model",
            },
            "OPENROUTER_API_KEY",
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
                "OPENROUTER_MODEL": "provider/model",
                "OPENROUTER_API_KEY": "key",
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
        LlmSettings.from_env(profile)
