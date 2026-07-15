"""Tests for planning and claim-research deep agents."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    ModelResponse,
    RunUsage,
    ToolCallPart,
    UsageLimitExceeded,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_core import to_jsonable_python

from knowledge_agent.agents.claim_researcher import (
    ClaimResearchPlan,
    ResearchClarification,
    run_claim_planning,
    run_claim_research,
)
from knowledge_agent.claims.store import load_claim_store
from knowledge_agent.claims.models import ClaimManifest
from knowledge_agent.llm.providers import AgentRuntime


SAMPLE_OUTPUT = "examples/claims/sample_output"
INVOICE_REF = "CLM-SAMPLE-001/DOC-002#DOC-002-CHUNK-001"


@contextmanager
def function_runtime(function):
    runner = asyncio.Runner()
    try:
        yield AgentRuntime(
            model=FunctionModel(function),
            runner=runner,
            openai=cast(Any, None),
        )
    finally:
        runner.close()


def _final(info: AgentInfo, *, answer: str, refs: list[str], sufficient: bool):
    return ModelResponse(
        parts=[
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "answer": answer,
                    "source_refs": refs,
                    "evidence_sufficient": sufficient,
                },
            )
        ]
    )


def _planning_output(info: AgentInfo, model_name: str, payload: dict[str, object]):
    tool_name = next(
        tool.name for tool in info.output_tools if model_name in tool.name
    )
    return ModelResponse(parts=[ToolCallPart(tool_name, payload)])


def test_planning_clarifies_then_returns_plan_in_one_native_context():
    calls = 0
    observed_tools = set()
    observed_message_counts = []

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        observed_message_counts.append(len(messages))
        observed_tools.update(tool.name for tool in info.function_tools)
        if calls == 1:
            return _planning_output(
                info,
                "ResearchClarification",
                {
                    "question": "Which repair period should be reviewed?",
                    "reason": "The requested time scope is ambiguous.",
                },
            )
        return _planning_output(
            info,
            "ClaimResearchPlan",
            {
                "objective": "Identify invoiced repairs.",
                "understood_scope": "Review all repair invoices.",
                "assumptions": ["All claim dates are in scope."],
                "searches": [
                    {
                        "query": "repair invoice",
                        "research_goal": "Identify invoiced repairs.",
                    }
                ],
                "completion_criteria": ["Each reported repair has a citation."],
            },
        )

    with function_runtime(model_function) as runtime:
        clarification = run_claim_planning(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "What was repaired?",
        )
        plan = run_claim_planning(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "Review all dates.",
            message_history=clarification.all_messages(),
            clarification_round=1,
        )

    assert isinstance(clarification.output, ResearchClarification)
    assert isinstance(plan.output, ClaimResearchPlan)
    assert observed_message_counts[1] > observed_message_counts[0]
    assert "claim_search" not in observed_tools
    assert {"read_todos", "write_todos", "add_todo"}.issubset(observed_tools)


def test_approved_research_continues_the_planning_message_history():
    def planning_model(messages, info: AgentInfo):
        return _planning_output(
            info,
            "ClaimResearchPlan",
            {
                "objective": "Identify repairs.",
                "understood_scope": "Review all invoices.",
                "assumptions": [],
                "searches": [
                    {
                        "query": "repair invoice",
                        "research_goal": "Find repaired items.",
                    }
                ],
                "completion_criteria": ["Cite repaired items."],
            },
        )

    with function_runtime(planning_model) as runtime:
        planned = run_claim_planning(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "What was repaired?",
        )

    research_calls = 0
    observed_message_counts = []

    def research_model(messages, info: AgentInfo):
        nonlocal research_calls
        research_calls += 1
        observed_message_counts.append(len(messages))
        if research_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "claim_search",
                        {"query": "repair invoice", "research_goal": "Find repairs"},
                    )
                ]
            )
        return _final(
            info,
            answer=f"A bumper was invoiced. [{INVOICE_REF}]",
            refs=[INVOICE_REF],
            sufficient=True,
        )

    with function_runtime(research_model) as runtime:
        report = run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "The plan is approved. Execute it.",
            approved_plan=planned.output,
            message_history=planned.all_messages(),
        )

    planning_messages = planned.all_messages()
    assert report.all_messages()[: len(planning_messages)] == planning_messages
    assert observed_message_counts[0] == len(planning_messages)
    assert report.output.source_refs == [INVOICE_REF]


def test_deep_agent_is_claim_scoped_and_returns_native_result():
    calls = 0
    observed_tools = set()

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        observed_tools.update(tool.name for tool in info.function_tools)
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "claim_search",
                        {
                            "query": "bumper invoice",
                            "research_goal": "Identify invoiced repairs",
                        },
                        tool_call_id="search-1",
                    )
                ]
            )
        return _final(
            info,
            answer=f"A bumper cover was invoiced. [{INVOICE_REF}]",
            refs=[INVOICE_REF],
            sufficient=True,
        )

    with function_runtime(model_function) as runtime:
        result = run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "What repairs were invoiced?",
        )

    assert result.output.source_refs == [INVOICE_REF]
    assert result.usage.requests == 2
    assert result.usage.tool_calls == 1
    assert "claim_search" in observed_tools
    assert {"read_todos", "write_todos", "add_todo"}.issubset(observed_tools)
    assert observed_tools.isdisjoint(
        {
            "read_file",
            "write_file",
            "execute",
            "web_search",
            "task",
            "read_memory",
        }
    )


def test_deep_agent_emits_native_tool_events():
    events = []
    runner = asyncio.Runner()
    runtime = AgentRuntime(
        model=TestModel(),
        runner=runner,
        openai=cast(Any, None),
    )
    try:
        run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "Inspect the claim.",
            on_event=events.append,
        )
    finally:
        runner.close()

    assert any(
        isinstance(event, FunctionToolCallEvent)
        and event.part.tool_name == "claim_search"
        for event in events
    )
    assert any(
        isinstance(event, FunctionToolResultEvent)
        and event.part.tool_name == "claim_search"
        for event in events
    )


def test_native_messages_round_trip_and_resume_chat_history():
    first_calls = 0

    def first_model(messages, info: AgentInfo):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "claim_search",
                        {"query": "bumper", "research_goal": "Find repairs"},
                    )
                ]
            )
        return _final(
            info,
            answer=f"A bumper was listed. [{INVOICE_REF}]",
            refs=[INVOICE_REF],
            sufficient=True,
        )

    with function_runtime(first_model) as runtime:
        first = run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "What was repaired?",
        )

    serialized = to_jsonable_python(first.all_messages())
    history = ModelMessagesTypeAdapter.validate_python(serialized)
    observed_message_counts = []

    def second_model(messages, info: AgentInfo):
        observed_message_counts.append(len(messages))
        return _final(
            info,
            answer="The claim does not establish who paid.",
            refs=[],
            sufficient=False,
        )

    with function_runtime(second_model) as runtime:
        second = run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "Who paid for it?",
            message_history=history,
        )

    assert observed_message_counts[0] > len(second.new_messages())
    assert not second.output.evidence_sufficient


def test_invalid_citation_gets_one_output_retry():
    calls = 0

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "claim_search",
                        {"query": "bumper", "research_goal": "Find repairs"},
                    )
                ]
            )
        if calls == 2:
            invalid = "CLM-SAMPLE-001/DOC-999#DOC-999-CHUNK-001"
            return _final(
                info,
                answer=f"Unsupported. [{invalid}]",
                refs=[invalid],
                sufficient=True,
            )
        return _final(
            info,
            answer=f"A bumper was listed. [{INVOICE_REF}]",
            refs=[INVOICE_REF],
            sufficient=True,
        )

    with function_runtime(model_function) as runtime:
        result = run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "What was repaired?",
        )

    assert calls == 3
    assert result.output.source_refs == [INVOICE_REF]


def test_claim_search_limit_is_hard():
    calls = 0

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "claim_search",
                    {
                        "query": f"query {calls}",
                        "research_goal": "Keep searching",
                    },
                )
            ]
        )

    with function_runtime(model_function) as runtime:
        with pytest.raises(UsageLimitExceeded, match="claim_search limit of 1"):
            run_claim_research(
                runtime,
                load_claim_store(SAMPLE_OUTPUT),
                "Search repeatedly.",
                max_searches=1,
            )


def test_model_request_limit_is_hard():
    calls = 0

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "claim_search",
                    {
                        "query": f"query {calls}",
                        "research_goal": "Keep searching",
                    },
                )
            ]
        )

    with function_runtime(model_function) as runtime:
        with pytest.raises(UsageLimitExceeded, match="request_limit"):
            run_claim_research(
                runtime,
                load_claim_store(SAMPLE_OUTPUT),
                "Search repeatedly.",
                max_searches=10,
                request_limit=2,
            )


def test_parallel_claim_search_calls_run_concurrently(monkeypatch):
    barrier = asyncio.Barrier(2)
    worker_tasks = set()
    calls = 0

    async def fake_search(store, query, top_k):
        worker_tasks.add(id(asyncio.current_task()))
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return []

    monkeypatch.setattr(
        "knowledge_agent.agents.claim_researcher.workflow.search_claim_evidence",
        fake_search,
    )

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "claim_search",
                        {"query": "one", "research_goal": "First gap"},
                    ),
                    ToolCallPart(
                        "claim_search",
                        {"query": "two", "research_goal": "Second gap"},
                    ),
                ]
            )
        return _final(
            info,
            answer="The claim lacks enough evidence.",
            refs=[],
            sufficient=False,
        )

    with function_runtime(model_function) as runtime:
        run_claim_research(
            runtime,
            load_claim_store(SAMPLE_OUTPUT),
            "Check two gaps.",
        )

    assert len(worker_tasks) == 2


def test_lightrag_keyword_usage_is_aggregated_with_deep_agent_usage():
    store = load_claim_store(SAMPLE_OUTPUT)
    manifest = ClaimManifest.model_validate(
        store.manifest.model_dump()
        | {
            "retrieval_mode": "lightrag",
            "embedding_provider": "nvidia",
            "embedding_model": "baai/bge-m3",
        }
    )

    class Retriever:
        usage = None

        def bind_usage(self, usage, limits):
            self.usage = usage

        def clear_usage(self):
            pass

        async def retrieve_chunk_ids(self, query, top_k):
            self.usage.incr(RunUsage(requests=1, input_tokens=7, output_tokens=3))
            return [store.chunks[0].chunk_id]

    lightrag_store = replace(
        store,
        manifest=manifest,
        retrieval_mode="lightrag",
        lightrag=cast(Any, Retriever()),
    )
    calls = 0

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "claim_search",
                        {"query": "collision", "research_goal": "Find evidence"},
                    )
                ]
            )
        ref = store.chunks[0].source_ref
        return _final(
            info,
            answer=f"Supported [{ref}]",
            refs=[ref],
            sufficient=True,
        )

    with function_runtime(model_function) as runtime:
        result = run_claim_research(
            runtime,
            lightrag_store,
            "What happened?",
        )

    assert result.usage.requests == 3
    assert result.usage.input_tokens >= 7
    assert result.usage.output_tokens >= 3


def test_lightrag_keyword_usage_counts_toward_request_limit():
    store = load_claim_store(SAMPLE_OUTPUT)
    manifest = ClaimManifest.model_validate(
        store.manifest.model_dump()
        | {
            "retrieval_mode": "lightrag",
            "embedding_provider": "nvidia",
            "embedding_model": "baai/bge-m3",
        }
    )

    class Retriever:
        usage = None

        def bind_usage(self, usage, limits):
            self.usage = usage

        def clear_usage(self):
            pass

        async def retrieve_chunk_ids(self, query, top_k):
            self.usage.incr(RunUsage(requests=1))
            return []

    calls = 0

    def model_function(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "claim_search",
                    {"query": f"query {calls}", "research_goal": "Find evidence"},
                )
            ]
        )

    with function_runtime(model_function) as runtime:
        with pytest.raises(UsageLimitExceeded, match="request_limit"):
            run_claim_research(
                runtime,
                replace(
                    store,
                    manifest=manifest,
                    lightrag=cast(Any, Retriever()),
                ),
                "Search until stopped.",
                request_limit=2,
            )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"max_searches": 0}, "max_searches must be at least 1"),
        ({"request_limit": 0}, "request_limit must be at least 1"),
    ],
)
def test_request_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_claim_research(
            cast(AgentRuntime, None),
            load_claim_store(SAMPLE_OUTPUT),
            "Question",
            **kwargs,
        )


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="question cannot be empty"):
        run_claim_research(
            cast(AgentRuntime, None),
            load_claim_store(SAMPLE_OUTPUT),
            "  ",
        )
