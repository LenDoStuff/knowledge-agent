import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from knowledge_agent.infrastructure import (
    LlmSettings,
    PortableResponsesClient,
    create_openai_runtime,
)
from research_agent.agent import run_claim_research
from research_agent.llm import ResearchResponsesLlm


load_dotenv()
SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claim_kb" / "sample_output"
)
GOLDEN_DATASET = Path(__file__).parents[2] / "evals" / "azure_research.json"


class CityAnswer(BaseModel):
    city: str
    country: str


def assert_city_contract(settings: LlmSettings) -> None:
    with PortableResponsesClient(settings, create_openai_runtime(settings)) as client:
        result = client.parse(
            "Return the requested city and country.",
            "Give the capital of France and its country.",
            CityAnswer,
        )
    assert result.city.casefold() == "paris"
    assert result.country.casefold() == "france"


@pytest.mark.live_openrouter
@pytest.mark.skipif(
    os.getenv("RUN_OPENROUTER_CONTRACT_TEST") != "1",
    reason="set RUN_OPENROUTER_CONTRACT_TEST=1 to call OpenRouter",
)
def test_openrouter_structured_output_contract():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key or not model:
        pytest.fail(
            "OPENROUTER_API_KEY and OPENROUTER_MODEL are required for the "
            "live contract test"
        )
    assert_city_contract(
        LlmSettings(
            mode="home",
            model=model,
            reasoning_effort="medium",
            openrouter_api_key=api_key,
            azure_ai_project_endpoint=None,
        )
    )


def azure_settings() -> LlmSettings:
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    model = os.getenv("AZURE_OPENAI_MODEL")
    if not endpoint or not model:
        pytest.fail(
            "AZURE_AI_PROJECT_ENDPOINT and AZURE_OPENAI_MODEL are required"
        )
    return LlmSettings(
        mode="work",
        model=model,
        reasoning_effort="medium",
        openrouter_api_key=None,
        azure_ai_project_endpoint=endpoint,
    )


@pytest.mark.live_azure
@pytest.mark.skipif(
    os.getenv("RUN_AZURE_CONTRACT_TEST") != "1",
    reason="set RUN_AZURE_CONTRACT_TEST=1 to call Azure",
)
def test_azure_structured_output_contract():
    assert_city_contract(azure_settings())


@pytest.mark.live_azure
@pytest.mark.skipif(
    os.getenv("RUN_AZURE_CONTRACT_TEST") != "1",
    reason="set RUN_AZURE_CONTRACT_TEST=1 to run the Azure golden evaluation",
)
def test_azure_research_golden_dataset():
    import json

    cases = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    settings = azure_settings()
    with PortableResponsesClient(
        settings,
        create_openai_runtime(settings),
    ) as client:
        llm = ResearchResponsesLlm(client)
        for case in cases:
            answer = run_claim_research(
                SAMPLE_OUTPUT,
                case["question"],
                llm,
                breadth=1,
                depth=1,
                top_k=2,
            )
            assert set(case["required_source_refs"]).issubset(answer.source_refs)
            answer_text = answer.answer.casefold()
            assert all(term in answer_text for term in case["required_terms"])
