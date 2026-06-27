"""Live dependency construction for claim research."""

from contextlib import contextmanager
from typing import Iterator

from infrastructure import (
    LlmSettings,
    PortableResponsesClient,
    create_openai_runtime,
)
from research.llm import ResearchResponsesLlm


@contextmanager
def live_research_llm() -> Iterator[ResearchResponsesLlm]:
    settings = LlmSettings.from_env()
    responses = PortableResponsesClient(
        settings,
        create_openai_runtime(settings),
    )
    try:
        yield ResearchResponsesLlm(responses)
    finally:
        responses.close()
