"""Claim-scoped Pydantic Deep Agents research workflow."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass, field
from threading import Lock

from pydantic_ai import (
    AgentRunResult,
    AgentStreamEvent,
    ModelMessage,
    ModelRetry,
    RunUsage,
    RunContext,
    UsageLimitExceeded,
    UsageLimits,
)
from pydantic_ai_summarization import create_sliding_window_processor
from pydantic_deep import DeepAgentDeps, StateBackend, create_deep_agent

from knowledge_agent.agents.claim_researcher.models import (
    ClaimResearchPlan,
    ClaimResearchOutput,
    EvidenceItem,
    ResearchClarification,
)
from knowledge_agent.agents.claim_researcher.prompts import (
    build_planning_instructions,
    build_research_instructions,
)
from knowledge_agent.agents.claim_researcher.tools import search_claim_evidence
from knowledge_agent.agents.claim_researcher.validation import validate_research_output
from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.llm.providers import AgentRuntime


LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_K = 8
DEFAULT_MAX_SEARCHES = 6
DEFAULT_REQUEST_LIMIT = 10
DEFAULT_PLANNING_REQUEST_LIMIT = 4
MAX_CLARIFICATIONS = 3
EventCallback = Callable[[AgentStreamEvent], None]
PlanningOutput = ResearchClarification | ClaimResearchPlan


@dataclass
class _SearchRun:
    store: ClaimStore
    top_k: int
    max_searches: int
    search_count: int = 0
    retrieved_source_refs: set[str] = field(default_factory=set)
    lock: Lock = field(default_factory=Lock)

    async def claim_search(
        self,
        query: str,
        research_goal: str,
    ) -> list[EvidenceItem]:
        """Search this claim for evidence relevant to a research goal."""

        query = query.strip()
        research_goal = research_goal.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not research_goal:
            raise ValueError("research_goal cannot be empty")
        with self.lock:
            if self.search_count >= self.max_searches:
                raise UsageLimitExceeded(
                    f"claim_search limit of {self.max_searches} calls exceeded"
                )
            self.search_count += 1

        evidence = await search_claim_evidence(self.store, query, self.top_k)
        with self.lock:
            self.retrieved_source_refs.update(item.source_ref for item in evidence)
        LOGGER.info(
            "claim_search query=%r goal=%r count=%d source_refs=%s",
            query,
            research_goal,
            len(evidence),
            [item.source_ref for item in evidence],
        )
        return evidence


def run_claim_research(
    runtime: AgentRuntime,
    store: ClaimStore,
    question: str,
    *,
    message_history: Sequence[ModelMessage] = (),
    top_k: int = DEFAULT_TOP_K,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    request_limit: int = DEFAULT_REQUEST_LIMIT,
    approved_plan: ClaimResearchPlan | None = None,
    on_event: EventCallback | None = None,
) -> AgentRunResult[ClaimResearchOutput]:
    question = _validate_research_request(
        question,
        top_k=top_k,
        max_searches=max_searches,
        request_limit=request_limit,
    )
    search_run = _SearchRun(store, top_k, max_searches)
    history_processor = create_sliding_window_processor(
        trigger=("messages", 40),
        keep=("messages", 20),
    )
    agent = create_deep_agent(
        model=runtime.model,
        instructions=build_research_instructions(store.documents, approved_plan),
        tools=[search_run.claim_search],
        output_type=ClaimResearchOutput,
        include_todo=True,
        include_filesystem=False,
        include_subagents=False,
        include_builtin_subagents=False,
        include_skills=False,
        include_plan=False,
        include_memory=False,
        include_checkpoints=False,
        include_teams=False,
        include_monitoring=False,
        include_improve=False,
        include_liteparse=False,
        include_history_archive=False,
        web_search=False,
        web_fetch=False,
        thinking=False,
        context_manager=False,
        history_processors=[history_processor],
        eviction_token_limit=None,
        cost_tracking=False,
        retries=1,
        stuck_loop_detection=True,
        patch_tool_calls=True,
        max_concurrency=4,
    )

    @agent.output_validator
    def validate_output(
        _ctx: RunContext[DeepAgentDeps],
        output: ClaimResearchOutput,
    ) -> ClaimResearchOutput:
        try:
            return validate_research_output(
                output,
                search_run.retrieved_source_refs,
            )
        except ValueError as exc:
            raise ModelRetry(str(exc)) from None

    event_stream_handler = _event_stream_handler(on_event)

    LOGGER.info(
        "research_start claim_id=%s question=%r history=%d top_k=%d "
        "max_searches=%d request_limit=%d",
        store.manifest.claim_id,
        question,
        len(message_history),
        top_k,
        max_searches,
        request_limit,
    )
    usage = RunUsage()
    usage_limits = UsageLimits(request_limit=request_limit)
    if store.lightrag is not None:
        store.lightrag.bind_usage(usage, usage_limits)
    try:
        result = runtime.run(
            agent,
            question,
            deps=DeepAgentDeps(backend=StateBackend()),
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
            event_stream_handler=event_stream_handler,
        )
    finally:
        if store.lightrag is not None:
            store.lightrag.clear_usage()
    LOGGER.info(
        "research_complete searches=%d sources=%d usage=%s",
        search_run.search_count,
        len(result.output.source_refs),
        result.usage,
    )
    return result


def run_claim_planning(
    runtime: AgentRuntime,
    store: ClaimStore,
    prompt: str,
    *,
    message_history: Sequence[ModelMessage] = (),
    clarification_round: int = 0,
    request_limit: int = DEFAULT_PLANNING_REQUEST_LIMIT,
    on_event: EventCallback | None = None,
) -> AgentRunResult[PlanningOutput]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("planning prompt cannot be empty")
    if clarification_round < 0 or clarification_round > MAX_CLARIFICATIONS:
        raise ValueError(f"clarification_round must be between 0 and {MAX_CLARIFICATIONS}")
    if request_limit < 1:
        raise ValueError("request_limit must be at least 1")

    agent = create_deep_agent(
        model=runtime.model,
        instructions=build_planning_instructions(
            store.documents,
            clarification_round=clarification_round,
            max_clarifications=MAX_CLARIFICATIONS,
        ),
        output_type=ResearchClarification | ClaimResearchPlan,
        include_todo=True,
        include_filesystem=False,
        include_subagents=False,
        include_builtin_subagents=False,
        include_skills=False,
        include_plan=False,
        include_memory=False,
        include_checkpoints=False,
        include_teams=False,
        include_monitoring=False,
        include_improve=False,
        include_liteparse=False,
        include_history_archive=False,
        web_search=False,
        web_fetch=False,
        thinking=False,
        context_manager=False,
        eviction_token_limit=None,
        cost_tracking=False,
        retries=1,
        stuck_loop_detection=True,
        patch_tool_calls=True,
    )

    @agent.output_validator
    def validate_planning_output(
        _ctx: RunContext[DeepAgentDeps],
        output: PlanningOutput,
    ) -> PlanningOutput:
        if (
            clarification_round >= MAX_CLARIFICATIONS
            and isinstance(output, ResearchClarification)
        ):
            raise ModelRetry(
                "The clarification limit was reached; return ClaimResearchPlan."
            )
        return output

    return runtime.run(
        agent,
        prompt,
        deps=DeepAgentDeps(backend=StateBackend()),
        message_history=message_history,
        usage_limits=UsageLimits(request_limit=request_limit),
        event_stream_handler=_event_stream_handler(on_event),
    )


def _event_stream_handler(on_event: EventCallback | None):
    if on_event is None:
        return None

    async def handle_events(
        _ctx: RunContext[DeepAgentDeps],
        events: AsyncIterable[AgentStreamEvent],
    ) -> None:
        async for event in events:
            on_event(event)

    return handle_events


def _validate_research_request(
    question: str,
    *,
    top_k: int,
    max_searches: int,
    request_limit: int,
) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_searches < 1:
        raise ValueError("max_searches must be at least 1")
    if request_limit < 1:
        raise ValueError("request_limit must be at least 1")
    return question
