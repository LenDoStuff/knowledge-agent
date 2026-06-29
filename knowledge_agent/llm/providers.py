"""OpenAI-compatible provider client construction."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential
from openai import OpenAI

from knowledge_agent.llm.config import LlmSettings


@dataclass(frozen=True)
class ProviderClients:
    openai: Any
    azure_project: Any | None = None
    azure_credential: Any | None = None


@contextmanager
def open_provider_clients(settings: LlmSettings) -> Iterator[ProviderClients]:
    with ExitStack() as stack:
        if settings.profile == "api_key":
            client = OpenAI(
                api_key=settings.nvidia_api_key_ds4,
                base_url=settings.nvidia_base_url,
                max_retries=0,
            )
            stack.callback(client.close)
            yield ProviderClients(openai=client)
            return

        credential = create_browser_credential()
        stack.callback(credential.close)
        project = AIProjectClient(
            endpoint=settings.azure_ai_project_endpoint,
            credential=credential,
        )
        stack.callback(project.close)
        client = project.get_openai_client().with_options(max_retries=0)
        stack.callback(client.close)
        yield ProviderClients(
            openai=client,
            azure_project=project,
            azure_credential=credential,
        )


def create_browser_credential() -> InteractiveBrowserCredential:
    return InteractiveBrowserCredential()
