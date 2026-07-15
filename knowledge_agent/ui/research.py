"""Streamlit state transitions for planning and claim research interactions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, cast

import streamlit as st
from pydantic_core import to_jsonable_python

from knowledge_agent.agents.claim_researcher import (
    ResearchClarification,
    run_claim_planning,
    run_claim_research,
)
from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import open_claim_store
from knowledge_agent.claims.lightrag import load_lightrag_metadata
from knowledge_agent.claims.models import RetrievalMode
from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import load_llm_settings
from knowledge_agent.llm.providers import open_agent_runtime
from knowledge_agent.research.history import (
    ClarificationExchange,
    InteractionUsage,
    KnowledgeBaseSnapshot,
    ResearchHistory,
    ResearchInteraction,
    load_research_history,
    store_interaction,
    utc_now,
)
from knowledge_agent.ui.claims import ClaimEntry
from knowledge_agent.ui.reports import (
    render_knowledge_base_snapshot,
    render_native_audit,
    render_plan,
    render_report_history,
    tool_query,
    tool_source_refs,
)


def render_research(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    local_store: ClaimStore,
) -> None:
    claim_id = entry.manifest.claim_id
    try:
        history = load_research_history(entry.path, claim_id)
    except Exception as exc:
        st.error(f"Could not load research history: {exc}")
        return

    st.subheader(f"Research {claim_id}")
    st.warning(
        "Saved reports and audits contain conversation text, model messages, "
        "and full claim evidence returned by tools."
    )
    new_tab, history_tab = st.tabs(["New research", "Report history"])
    with new_tab:
        _render_new_research(entry, claim_settings, local_store, history)
    with history_tab:
        render_report_history(entry, local_store, history)


def _render_new_research(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    store: ClaimStore,
    history: ResearchHistory,
) -> None:
    claim_id = entry.manifest.claim_id
    active = history.active_interaction()
    selected_mode: RetrievalMode | None = None
    if active is None:
        selected_mode = cast(
            RetrievalMode,
            st.selectbox(
                "Research knowledge base",
                list(entry.manifest.available_retrieval_modes),
                format_func=_retrieval_mode_label,
                key=f"research_engine_{claim_id}",
                help="This choice is saved with the report and cannot change mid-run.",
            ),
        )
    planning_column, audit_column = st.columns(2)
    planning_enabled = planning_column.toggle(
        "Planning",
        value=True,
        key=f"research_planning_{claim_id}",
        help="Clarify scope and require approval before searching.",
    )
    show_live_audit = audit_column.toggle(
        "Show live audit",
        value=False,
        key=f"research_live_audit_{claim_id}",
    )

    if active is not None:
        _render_active_interaction(
            entry,
            claim_settings,
            store,
            active,
            show_live_audit,
        )
        return

    prompt = st.chat_input(
        "Ask a question about this claim",
        key=f"new_research_prompt_{claim_id}",
    )
    if not prompt:
        st.caption("Each submitted question starts an independent research context.")
        return
    if is_greeting(prompt):
        st.info("Ask a specific question about the selected claim.")
        return

    try:
        if selected_mode is None:
            raise ValueError("No research knowledge base was selected")
        knowledge_base = _knowledge_base_snapshot(entry, selected_mode)
    except Exception as exc:
        st.error(f"Could not validate the selected knowledge base: {exc}")
        return

    interaction = ResearchInteraction(
        claim_id=claim_id,
        status="planning" if planning_enabled else "researching",
        question=prompt.strip(),
        planning_enabled=planning_enabled,
        knowledge_base=knowledge_base,
    )
    store_interaction(entry.path, interaction)
    if planning_enabled:
        _run_planning_phase(entry, store, interaction, prompt)
    else:
        _run_research_phase(entry, claim_settings, interaction, prompt)
    st.rerun()


def _render_active_interaction(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    store: ClaimStore,
    interaction: ResearchInteraction,
    show_live_audit: bool,
) -> None:
    with st.chat_message("user"):
        st.markdown(interaction.question)
    render_knowledge_base_snapshot(interaction.knowledge_base)
    for exchange in interaction.clarifications:
        with st.chat_message("assistant"):
            st.markdown(exchange.question)
            st.caption(exchange.reason)
        if exchange.answer is not None:
            with st.chat_message("user"):
                st.markdown(exchange.answer)

    if interaction.status == "awaiting_clarification":
        answer = st.chat_input(
            "Answer the clarification question",
            key=f"clarification_{interaction.id}",
        )
        if answer:
            pending = interaction.clarifications[-1]
            interaction.clarifications[-1] = ClarificationExchange(
                question=pending.question,
                reason=pending.reason,
                answer=answer.strip(),
            )
            interaction.status = "planning"
            interaction.updated_at = utc_now()
            store_interaction(entry.path, interaction)
            _run_planning_phase(entry, store, interaction, answer)
            st.rerun()
    elif interaction.status == "awaiting_approval":
        st.markdown("### Proposed research plan")
        render_plan(interaction.plan)
        approve, cancel = st.columns(2)
        if approve.button(
            "Approve and run",
            key=f"approve_{interaction.id}",
            type="primary",
            width="stretch",
        ):
            interaction.status = "researching"
            interaction.updated_at = utc_now()
            store_interaction(entry.path, interaction)
            _run_research_phase(
                entry,
                claim_settings,
                interaction,
                "The proposed research plan is approved. Execute it and produce the report.",
            )
            st.rerun()
        if cancel.button(
            "Cancel",
            key=f"cancel_{interaction.id}",
            width="stretch",
        ):
            interaction.status = "cancelled"
            interaction.updated_at = utc_now()
            store_interaction(entry.path, interaction)
            st.rerun()
    elif interaction.status in {"planning", "researching"}:
        st.warning(
            "This interaction was interrupted while an agent call was running. "
            "Retry it or cancel it."
        )
        retry, cancel = st.columns(2)
        if retry.button("Retry", key=f"retry_{interaction.id}", width="stretch"):
            if interaction.status == "planning":
                prompt = (
                    interaction.clarifications[-1].answer
                    if interaction.clarifications
                    else interaction.question
                )
                _run_planning_phase(
                    entry,
                    store,
                    interaction,
                    prompt or interaction.question,
                )
            else:
                prompt = (
                    "The proposed research plan is approved. Execute it and produce "
                    "the report."
                    if interaction.plan is not None
                    else interaction.question
                )
                _run_research_phase(entry, claim_settings, interaction, prompt)
            st.rerun()
        if cancel.button(
            "Cancel",
            key=f"cancel_interrupted_{interaction.id}",
            width="stretch",
        ):
            interaction.status = "cancelled"
            interaction.updated_at = utc_now()
            store_interaction(entry.path, interaction)
            st.rerun()

    if show_live_audit:
        render_native_audit(interaction.agent_messages, interaction.audit_events)


def _run_planning_phase(
    entry: ClaimEntry,
    store: ClaimStore,
    interaction: ResearchInteraction,
    prompt: str,
) -> None:
    remaining_requests = 4 - interaction.planning_usage.requests
    if remaining_requests < 1:
        _fail_interaction(entry.path, interaction, "planning request limit of 4 exceeded")
        return
    events: list[dict[str, object]] = []
    try:
        settings = load_llm_settings(load_profile())
        with st.status("Planning the research…", expanded=True) as status:
            status.write("Verifying the question and its scope.")
            with open_agent_runtime(settings) as runtime:
                result = run_claim_planning(
                    runtime,
                    store,
                    prompt,
                    message_history=interaction.agent_messages,
                    clarification_round=len(interaction.clarifications),
                    request_limit=remaining_requests,
                    on_event=_audit_callback(events, "planning"),
                )
            _write_event_status(status, events)
            status.update(label="Planning step complete", state="complete")
        interaction.agent_messages += result.new_messages()
        interaction.audit_events.extend(events)
        interaction.planning_usage = interaction.planning_usage.plus(
            _interaction_usage(result.usage)
        )
        if isinstance(result.output, ResearchClarification):
            interaction.clarifications.append(
                ClarificationExchange(
                    question=result.output.question,
                    reason=result.output.reason,
                )
            )
            interaction.status = "awaiting_clarification"
        else:
            interaction.plan = result.output
            interaction.status = "awaiting_approval"
        interaction.updated_at = utc_now()
        store_interaction(entry.path, interaction)
    except Exception as exc:
        interaction.audit_events.extend(events)
        _fail_interaction(entry.path, interaction, str(exc) or exc.__class__.__name__)


def _run_research_phase(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    interaction: ResearchInteraction,
    prompt: str,
) -> None:
    events: list[dict[str, object]] = []
    try:
        if interaction.knowledge_base is None:
            raise ValueError(
                "This active interaction has no knowledge-base snapshot and cannot "
                "be resumed. Cancel it and start a new research interaction."
            )
        settings = load_llm_settings(load_profile())
        retrieval_mode = interaction.knowledge_base.retrieval_mode
        with st.status("Researching the claim…", expanded=True) as status:
            status.write(
                f"Searching {_retrieval_mode_label(retrieval_mode)} and preparing "
                "the report."
            )
            with (
                open_agent_runtime(settings) as runtime,
                open_claim_store(
                    entry.path,
                    claim_settings,
                    retrieval_mode=retrieval_mode,
                    runtime=runtime,
                    llm_settings=settings,
                ) as store,
            ):
                result = run_claim_research(
                    runtime,
                    store,
                    prompt,
                    message_history=interaction.agent_messages,
                    approved_plan=interaction.plan,
                    on_event=_audit_callback(events, "research"),
                )
            _write_event_status(status, events)
            status.update(label="Research complete", state="complete")
        interaction.agent_messages += result.new_messages()
        interaction.audit_events.extend(events)
        interaction.research_usage = interaction.research_usage.plus(
            _interaction_usage(result.usage)
        )
        interaction.output = result.output
        interaction.status = "completed"
        interaction.updated_at = utc_now()
        store_interaction(entry.path, interaction)
    except Exception as exc:
        interaction.audit_events.extend(events)
        _fail_interaction(entry.path, interaction, str(exc) or exc.__class__.__name__)


def _audit_callback(events: list[dict[str, object]], phase: str):
    def capture(event: object) -> None:
        events.append(
            {
                "phase": phase,
                "type": event.__class__.__name__,
                "payload": to_jsonable_python(event),
            }
        )

    return capture


def _write_event_status(status, events: Sequence[dict[str, object]]) -> None:
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        part = payload.get("part")
        if not isinstance(part, dict) or part.get("tool_name") != "claim_search":
            continue
        if event.get("type") == "FunctionToolCallEvent":
            status.write(f"Searched claim evidence: {tool_query(part.get('args'))}")
        elif event.get("type") == "FunctionToolResultEvent":
            status.write(
                "Claim search returned "
                f"{len(tool_source_refs(part.get('content')))} chunks."
            )


def _fail_interaction(
    claim_path: Path,
    interaction: ResearchInteraction,
    error: str,
) -> None:
    interaction.error = error
    interaction.status = "failed"
    interaction.updated_at = utc_now()
    store_interaction(claim_path, interaction)


def _interaction_usage(usage: object) -> InteractionUsage:
    return InteractionUsage(**_usage_payload(usage))


def _usage_payload(usage: object) -> dict[str, int]:
    return {
        "requests": int(getattr(usage, "requests", 0)),
        "tool_calls": int(getattr(usage, "tool_calls", 0)),
        "input_tokens": int(getattr(usage, "input_tokens", 0)),
        "output_tokens": int(getattr(usage, "output_tokens", 0)),
    }


def _knowledge_base_snapshot(
    entry: ClaimEntry,
    retrieval_mode: RetrievalMode,
) -> KnowledgeBaseSnapshot:
    manifest = entry.manifest
    if retrieval_mode not in manifest.available_retrieval_modes:
        raise ValueError(
            f"Retrieval mode {retrieval_mode!r} is not available for this claim"
        )
    if retrieval_mode != "lightrag":
        return KnowledgeBaseSnapshot(
            retrieval_mode=retrieval_mode,
            embedding_provider=(
                manifest.embedding_provider if retrieval_mode == "semantic" else None
            ),
            embedding_model=(
                manifest.embedding_model if retrieval_mode == "semantic" else None
            ),
        )
    metadata = load_lightrag_metadata(entry.path / "index" / "lightrag")
    if metadata.claim_id != manifest.claim_id:
        raise ValueError("LightRAG metadata claim_id does not match the claim")
    return KnowledgeBaseSnapshot(
        retrieval_mode=retrieval_mode,
        embedding_provider=manifest.embedding_provider,
        embedding_model=manifest.embedding_model,
        lightrag_version=metadata.lightrag_version,
        lightrag_index_claim_id=metadata.claim_id,
        lightrag_index_llm_provider=metadata.llm_provider,
        lightrag_index_llm_model=metadata.llm_model,
        lightrag_embedding_dimension=metadata.embedding_dimension,
        lightrag_embedding_max_tokens=metadata.embedding_max_tokens,
        lightrag_query_mode=metadata.query_mode,
        lightrag_indexed_chunk_count=metadata.indexed_chunk_count,
        lightrag_entity_count=metadata.entity_count,
        lightrag_relationship_count=metadata.relationship_count,
        lightrag_indexing_usage=InteractionUsage(
            **metadata.indexing_usage.model_dump()
        ),
        lightrag_index_created_at=metadata.created_at,
    )


def _retrieval_mode_label(retrieval_mode: RetrievalMode) -> str:
    if retrieval_mode == "lexical":
        return "Custom (lexical)"
    if retrieval_mode == "semantic":
        return "Custom (semantic)"
    return "LightRAG"


def is_greeting(value: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", value.casefold()).strip()
    return normalized in {"hi", "hello", "hey", "good morning", "good evening"}
