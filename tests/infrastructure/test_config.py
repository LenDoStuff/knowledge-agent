import pytest

from infrastructure.config import ConfigurationError, LlmSettings


MODE_ENV_NAMES = [
    "KNOWLEDGE_AGENT_MODE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "AZURE_AI_PROJECT_ENDPOINT",
    "AZURE_OPENAI_MODEL",
    "LLM_REASONING_EFFORT",
]


def set_environment(monkeypatch, values):
    monkeypatch.setattr(
        "infrastructure.config.load_dotenv",
        lambda: None,
    )
    for name in MODE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_home_settings_load_without_exposing_secret(monkeypatch):
    set_environment(
        monkeypatch,
        {
            "KNOWLEDGE_AGENT_MODE": "home",
            "OPENROUTER_MODEL": "provider/model",
            "OPENROUTER_API_KEY": "secret-test-key",
            "LLM_REASONING_EFFORT": "high",
        },
    )

    settings = LlmSettings.from_env()

    assert settings.mode == "home"
    assert settings.provider == "openrouter"
    assert settings.model == "provider/model"
    assert settings.reasoning_effort == "high"
    assert "secret-test-key" not in repr(settings)
    assert "secret-test-key" not in str(settings.safe_summary())


def test_work_settings_select_azure_model(monkeypatch):
    set_environment(
        monkeypatch,
        {
            "KNOWLEDGE_AGENT_MODE": "work",
            "AZURE_OPENAI_MODEL": "deployment-name",
            "AZURE_AI_PROJECT_ENDPOINT": (
                "https://example.services.ai.azure.com/api/projects/project"
            ),
        },
    )

    settings = LlmSettings.from_env()

    assert settings.mode == "work"
    assert settings.provider == "azure"
    assert settings.model == "deployment-name"
    assert settings.openrouter_api_key is None


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "KNOWLEDGE_AGENT_MODE"),
        ({"KNOWLEDGE_AGENT_MODE": "unknown"}, "KNOWLEDGE_AGENT_MODE"),
        (
            {
                "KNOWLEDGE_AGENT_MODE": "home",
                "OPENROUTER_MODEL": "provider/model",
            },
            "OPENROUTER_API_KEY",
        ),
        (
            {
                "KNOWLEDGE_AGENT_MODE": "home",
                "OPENROUTER_API_KEY": "key",
            },
            "OPENROUTER_MODEL",
        ),
        (
            {
                "KNOWLEDGE_AGENT_MODE": "work",
                "AZURE_OPENAI_MODEL": "deployment",
            },
            "AZURE_AI_PROJECT_ENDPOINT",
        ),
        (
            {
                "KNOWLEDGE_AGENT_MODE": "work",
                "AZURE_AI_PROJECT_ENDPOINT": "https://project.example",
            },
            "AZURE_OPENAI_MODEL",
        ),
        (
            {
                "KNOWLEDGE_AGENT_MODE": "home",
                "OPENROUTER_MODEL": "provider/model",
                "OPENROUTER_API_KEY": "key",
                "LLM_REASONING_EFFORT": "extreme",
            },
            "LLM_REASONING_EFFORT",
        ),
    ],
)
def test_invalid_mode_configuration_fails_at_startup(
    monkeypatch,
    environment,
    message,
):
    set_environment(monkeypatch, environment)

    with pytest.raises(ConfigurationError, match=message):
        LlmSettings.from_env()
