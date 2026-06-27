"""OpenAI-compatible client construction and resource ownership."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential
from openai import OpenAI

from infrastructure.config import ConfigurationError, LlmSettings


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAiRuntime(AbstractContextManager["OpenAiRuntime"]):
    def __init__(
        self,
        client: Any,
        resources: list[Any],
        project_client: Any | None = None,
        azure_credential: Any | None = None,
    ) -> None:
        self.client = client
        self.project_client = project_client
        self.azure_credential = azure_credential
        self._resources = resources
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for resource in self._resources:
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_openai_runtime(settings: LlmSettings) -> OpenAiRuntime:
    if settings.mode == "home":
        if settings.openrouter_api_key is None:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required in home mode"
            )
        client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            max_retries=0,
        )
        return OpenAiRuntime(client, [client])

    if settings.mode == "work":
        if settings.azure_ai_project_endpoint is None:
            raise ConfigurationError(
                "AZURE_AI_PROJECT_ENDPOINT is required in work mode"
            )
        credential = create_browser_credential()
        project = None
        try:
            project = AIProjectClient(
                endpoint=settings.azure_ai_project_endpoint,
                credential=credential,
            )
            client = project.get_openai_client().with_options(max_retries=0)
        except Exception:
            if project is not None:
                try:
                    project.close()
                finally:
                    credential.close()
            else:
                credential.close()
            raise
        return OpenAiRuntime(
            client,
            [client, project, credential],
            project_client=project,
            azure_credential=credential,
        )

    raise ConfigurationError(f"Unsupported runtime mode: {settings.mode}")


def create_browser_credential() -> InteractiveBrowserCredential:
    return InteractiveBrowserCredential()
