"""Opt-in live contracts for configured LLM providers."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

from knowledge_agent.agents.claim_researcher import run_claim_research
from knowledge_agent.claims.ocr import SnowflakeParseDocumentOcrClient
from knowledge_agent.claims.store import load_claim_store
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.llm.providers import open_agent_runtime


load_dotenv()
SAMPLE_OUTPUT = (
    Path(__file__).parents[2] / "examples" / "claims" / "sample_output"
)
GOLDEN_DATASET = Path(__file__).parents[2] / "evals" / "azure_research.json"
SAMPLE_PDF = (
    Path(__file__).parents[2]
    / "examples"
    / "claims"
    / "sample_input"
    / "00_claim_file_index.pdf"
)


class CityAnswer(BaseModel):
    city: str
    country: str


def assert_city_contract(settings: LlmSettings) -> None:
    agent = Agent(
        output_type=CityAnswer,
        instructions="Return the requested city and country.",
        retries={"tools": 1, "output": 1},
    )
    with open_agent_runtime(settings) as runtime:
        result = runtime.run(
            agent,
            "Give the capital of France and its country.",
        )
    assert result.output.city.casefold() == "paris"
    assert result.output.country.casefold() == "france"


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


def snowflake_settings() -> LlmSettings:
    model = os.getenv("SNOWFLAKE_CORTEX_MODEL")
    pat = os.getenv("SNOWFLAKE_CORTEX_PAT")
    if not model or not pat:
        pytest.fail("SNOWFLAKE_CORTEX_MODEL and SNOWFLAKE_CORTEX_PAT are required")
    return LlmSettings(
        profile="snowflake",
        model=model,
        reasoning_effort="medium",
        snowflake_connection_name=os.getenv("SNOWFLAKE_CONNECTION_NAME", "default"),
        snowflake_cortex_pat=pat,
    )


@pytest.mark.live_azure
@pytest.mark.skipif(
    os.getenv("RUN_AZURE_CONTRACT_TEST") != "1",
    reason="set RUN_AZURE_CONTRACT_TEST=1 to call Azure",
)
def test_azure_structured_output_contract():
    assert_city_contract(azure_settings())


@pytest.mark.live_snowflake
@pytest.mark.skipif(
    os.getenv("RUN_SNOWFLAKE_CONTRACT_TEST") != "1",
    reason="set RUN_SNOWFLAKE_CONTRACT_TEST=1 to call Snowflake Cortex",
)
def test_snowflake_structured_output_contract():
    assert_city_contract(snowflake_settings())


@pytest.mark.live_snowflake
@pytest.mark.skipif(
    os.getenv("RUN_SNOWFLAKE_CONTRACT_TEST") != "1",
    reason="set RUN_SNOWFLAKE_CONTRACT_TEST=1 to call Snowflake OCR",
)
def test_snowflake_ocr_contract(monkeypatch):
    run_id = "contract-ocr"
    monkeypatch.setattr(
        "knowledge_agent.claims.ocr.uuid4",
        lambda: SimpleNamespace(hex=run_id),
    )
    stage = os.getenv("SNOWFLAKE_DOCUMENT_STAGE", "KNOWLEDGE_AGENT_DOCUMENTS")
    with open_agent_runtime(snowflake_settings()) as runtime:
        assert runtime.snowflake_session is not None
        client = SnowflakeParseDocumentOcrClient(runtime.snowflake_session, stage)
        pages = client.extract_pages("SNOWFLAKE-CONTRACT", SAMPLE_PDF)
        remaining = runtime.snowflake_session.sql(
            f"LIST @{stage}/knowledge-agent/{run_id}"
        ).collect()

    assert pages
    assert pages[0].page_number == 1
    assert remaining == []


@pytest.mark.live_azure
@pytest.mark.skipif(
    os.getenv("RUN_AZURE_CONTRACT_TEST") != "1",
    reason="set RUN_AZURE_CONTRACT_TEST=1 to run the Azure golden evaluation",
)
def test_azure_research_golden_dataset():
    assert_research_golden(azure_settings())


def assert_research_golden(settings: LlmSettings) -> None:
    import json

    cases = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    with open_agent_runtime(settings) as runtime:
        store = load_claim_store(SAMPLE_OUTPUT)
        for case in cases:
            result = run_claim_research(
                runtime=runtime,
                store=store,
                question=case["question"],
                top_k=2,
            )
            assert set(case["required_source_refs"]).issubset(
                result.output.source_refs
            )
            answer_text = result.output.answer.casefold()
            assert all(term in answer_text for term in case["required_terms"])


@pytest.mark.live_nvidia
@pytest.mark.skipif(
    os.getenv("RUN_NVIDIA_CONTRACT_TEST") != "1",
    reason="set RUN_NVIDIA_CONTRACT_TEST=1 to run NVIDIA research evaluation",
)
def test_nvidia_research_golden_dataset():
    api_key = os.getenv("nvidia_api_key_ds4")
    base_url = os.getenv("nvidia_base_url")
    if not api_key or not base_url:
        pytest.fail("NVIDIA credentials are required")
    assert_research_golden(
        LlmSettings(
            profile="api_key",
            model="deepseek-ai/deepseek-v4-pro",
            reasoning_effort="medium",
            nvidia_base_url=base_url,
            nvidia_api_key_ds4=api_key,
        )
    )
