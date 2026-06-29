import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.llm.client import open_responses_client
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.research.agent import run_claim_research
from knowledge_agent.research.llm import ResponsesResearchModel


load_dotenv()
SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
)
GOLDEN_DATASET = Path(__file__).parents[2] / "evals" / "azure_research.json"


class CityAnswer(BaseModel):
    city: str
    country: str


def assert_city_contract(settings: LlmSettings) -> None:
    with open_responses_client(settings) as client:
        result = client.parse(
            "Return the requested city and country.",
            "Give the capital of France and its country.",
            CityAnswer,
        )
    assert result.city.casefold() == "paris"
    assert result.country.casefold() == "france"


@pytest.mark.live_nvidia
@pytest.mark.skipif(
    os.getenv("RUN_NVIDIA_CONTRACT_TEST") != "1",
    reason="set RUN_NVIDIA_CONTRACT_TEST=1 to call NVIDIA",
)
def test_nvidia_structured_output_contract():
    api_key = os.getenv("nvidia_api_key_ds4")
    base_url = os.getenv("nvidia_base_url")
    if not api_key or not base_url:
        pytest.fail(
            "nvidia_api_key_ds4 and nvidia_base_url are required for the "
            "live contract test"
        )
    assert_city_contract(
        LlmSettings(
            profile="api_key",
            model="deepseek-ai/deepseek-v4-pro",
            reasoning_effort="medium",
            nvidia_base_url=base_url,
            nvidia_api_key_ds4=api_key,
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
        profile="azure_project",
        model=model,
        reasoning_effort="medium",
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
    with open_responses_client(settings) as client:
        model = ResponsesResearchModel(client)
        store = ClaimStore(SAMPLE_OUTPUT)
        for case in cases:
            answer = run_claim_research(
                store=store,
                question=case["question"],
                model=model,
                queries_per_question=1,
                max_depth=1,
                top_k=2,
            )
            assert set(case["required_source_refs"]).issubset(answer.source_refs)
            answer_text = answer.answer.casefold()
            assert all(term in answer_text for term in case["required_terms"])
