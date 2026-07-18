"""PydanticAI model construction and synchronous runtime ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar, cast

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential, get_bearer_token_provider
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

from knowledge_agent.llm.config import LlmSettings


if TYPE_CHECKING:
    from snowflake.snowpark import Session


AZURE_AI_SCOPE = "https://ai.azure.com/.default"
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class AgentRuntime:
    """Own one model client and event loop for synchronous application code."""

    model: Model
    runner: asyncio.Runner
    openai: AsyncOpenAI
    loop: asyncio.AbstractEventLoop | None = None
    thread: Thread | None = None
    azure_project: AIProjectClient | None = None
    azure_credential: InteractiveBrowserCredential | None = None
    snowflake_session: Session | None = None

    def run(self, agent: Agent[Any, Any], prompt: str, **kwargs: Any) -> Any:
        """Run a PydanticAI agent on this runtime's persistent event loop."""

        coroutine = agent.run(prompt, model=self.model, **kwargs)
        return self.run_coroutine(coroutine)

    def run_coroutine(
        self,
        coroutine: Coroutine[Any, Any, ResultT],
    ) -> ResultT:
        """Run an arbitrary coroutine on the runtime's persistent event loop."""

        if self.loop is None:
            return self.runner.run(coroutine)
        future: Future[ResultT] = asyncio.run_coroutine_threadsafe(
            coroutine,
            self.loop,
        )
        return future.result()


@contextmanager
def open_agent_runtime(settings: LlmSettings) -> Iterator[AgentRuntime]:
    """Open the configured PydanticAI model and all provider resources."""

    runner = asyncio.Runner()
    loop = runner.get_loop()

    def run_loop() -> None:
        loop.run_forever()
        runner.close()

    thread = Thread(target=run_loop, name="pydantic-ai-runtime", daemon=True)
    thread.start()
    credential: InteractiveBrowserCredential | None = None
    project: AIProjectClient | None = None
    snowflake_session: Session | None = None
    client: AsyncOpenAI | None = None
    try:
        if settings.profile == "api_key":
            client = AsyncOpenAI(
                api_key=settings.nvidia_api_key_ds4,
                base_url=settings.nvidia_base_url,
                max_retries=0,
            )
            model = OpenAIChatModel(
                settings.model,
                provider=OpenAIProvider(openai_client=client),
                profile=deepseek_model_profile(settings.model.rsplit("/", 1)[-1]),
                settings=OpenAIChatModelSettings(
                    temperature=0,
                    thinking=False,
                    extra_body={
                        "chat_template_kwargs": {"thinking": False},
                    },
                ),
            )
        elif settings.profile == "azure_project":
            credential = create_browser_credential()
            project = AIProjectClient(
                endpoint=settings.azure_ai_project_endpoint,
                credential=credential,
            )
            token_provider = get_bearer_token_provider(credential, AZURE_AI_SCOPE)

            async def async_token_provider() -> str:
                return await asyncio.to_thread(token_provider)

            project_endpoint = str(settings.azure_ai_project_endpoint).rstrip("/")
            client = AsyncOpenAI(
                base_url=f"{project_endpoint}/openai/v1",
                api_key=async_token_provider,
                max_retries=0,
            )
            model = OpenAIResponsesModel(
                settings.model,
                provider=OpenAIProvider(openai_client=client),
                settings=OpenAIResponsesModelSettings(
                    openai_reasoning_effort=settings.reasoning_effort,
                ),
            )
        else:
            connection_name = cast(str, settings.snowflake_connection_name)
            snowflake_session = create_snowflake_session(connection_name)
            host = str(snowflake_session.connection.host).strip().rstrip("/")
            if not host:
                raise ValueError(
                    "The Snowflake named connection did not provide an account host"
                )
            client = AsyncOpenAI(
                base_url=f"https://{host}/api/v2/cortex/v1",
                api_key=cast(str, settings.snowflake_cortex_pat),
                default_headers={
                    "X-Snowflake-Authorization-Token-Type": (
                        "PROGRAMMATIC_ACCESS_TOKEN"
                    )
                },
                max_retries=0,
            )
            model = OpenAIChatModel(
                settings.model,
                provider=OpenAIProvider(openai_client=client),
                settings=OpenAIChatModelSettings(temperature=0),
            )

        yield AgentRuntime(
            model=model,
            runner=runner,
            loop=loop,
            thread=thread,
            openai=client,
            azure_project=project,
            azure_credential=credential,
            snowflake_session=snowflake_session,
        )
    finally:
        if client is not None:
            asyncio.run_coroutine_threadsafe(client.close(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        if project is not None:
            project.close()
        if snowflake_session is not None:
            snowflake_session.close()
        if credential is not None:
            credential.close()


def create_browser_credential() -> InteractiveBrowserCredential:
    return InteractiveBrowserCredential()


def create_snowflake_session(connection_name: str) -> Session:
    """Open one Snowpark session from a native named connection."""

    from snowflake.snowpark import Session

    return Session.builder.config("connection_name", connection_name).create()
