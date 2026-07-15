"""Streamlit rendering and trace parsing for saved research reports."""

from __future__ import annotations

import html
import re
from typing import Sequence

import streamlit as st
from pydantic_ai import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_core import to_jsonable_python

from knowledge_agent.agents.claim_researcher import (
    ClaimResearchOutput,
    ClaimResearchPlan,
)
from knowledge_agent.claims.errors import ChunkNotFoundError
from knowledge_agent.claims.models import DocumentChunk
from knowledge_agent.claims.store import ClaimStore, get_document
from knowledge_agent.research.history import (
    TERMINAL_STATUSES,
    KnowledgeBaseSnapshot,
    ResearchHistory,
    ResearchInteraction,
    clear_terminal_interactions,
    delete_interaction,
)
from knowledge_agent.ui.claims import ClaimEntry


def render_knowledge_base_snapshot(
    snapshot: KnowledgeBaseSnapshot | None,
) -> None:
    if snapshot is None:
        st.caption("Knowledge base: engine not recorded")
        return
    details = [snapshot.retrieval_mode]
    if snapshot.embedding_model:
        details.append(snapshot.embedding_model)
    if snapshot.lightrag_version:
        details.append(f"LightRAG {snapshot.lightrag_version}")
    if snapshot.lightrag_index_llm_model:
        details.append(f"indexed by {snapshot.lightrag_index_llm_model}")
    if snapshot.lightrag_index_created_at:
        details.append(
            "index "
            + snapshot.lightrag_index_created_at.astimezone().strftime(
                "%Y-%m-%d %H:%M"
            )
        )
    st.caption("Knowledge base: " + " · ".join(details))
    with st.expander("Knowledge base snapshot"):
        st.json(snapshot.model_dump(mode="json", exclude_none=True))


def render_plan(plan: ClaimResearchPlan | None) -> None:
    if plan is None:
        st.error("The saved interaction has no research plan.")
        return
    with st.container(border=True):
        st.markdown(f"**Objective:** {plan.objective}")
        st.markdown(f"**Scope:** {plan.understood_scope}")
        if plan.assumptions:
            st.markdown("**Assumptions**")
            for assumption in plan.assumptions:
                st.markdown(f"- {assumption}")
        st.markdown("**Planned searches**")
        for index, search in enumerate(plan.searches, start=1):
            st.markdown(f"{index}. `{search.query}` — {search.research_goal}")
        st.markdown("**Completion criteria**")
        for criterion in plan.completion_criteria:
            st.markdown(f"- {criterion}")


def render_report_history(
    entry: ClaimEntry,
    store: ClaimStore,
    history: ResearchHistory,
) -> None:
    if not history.interactions:
        st.info("No saved research interactions for this claim.")
        return
    interactions = sorted(
        history.interactions,
        key=lambda item: item.created_at,
        reverse=True,
    )
    by_id = {item.id: item for item in interactions}
    selected_id = st.selectbox(
        "Saved interaction",
        list(by_id),
        format_func=lambda interaction_id: _interaction_label(by_id[interaction_id]),
        key=f"history_select_{entry.manifest.claim_id}",
    )
    interaction = by_id[selected_id]
    st.caption(
        f"Status: {interaction.status} · Created: "
        f"{interaction.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.markdown(f"### {interaction.question}")
    render_knowledge_base_snapshot(interaction.knowledge_base)
    if interaction.clarifications:
        with st.expander(f"Clarifications ({len(interaction.clarifications)})"):
            for exchange in interaction.clarifications:
                st.markdown(f"**Agent:** {exchange.question}")
                st.markdown(f"**User:** {exchange.answer or '(unanswered)'}")
    if interaction.plan is not None:
        st.markdown("### Approved plan")
        render_plan(interaction.plan)
    if interaction.output is not None:
        st.markdown("### Report")
        try:
            _render_cited_answer(interaction.output, store)
            _render_research_details(
                interaction.output,
                interaction.agent_messages,
                interaction.usage.model_dump(),
                store,
            )
        except Exception as exc:
            st.error(f"Could not display saved research report: {exc}")
    if interaction.error:
        st.error(interaction.error)
    if interaction.status == "cancelled":
        st.info("This interaction was cancelled before a report was produced.")
    render_native_audit(interaction.agent_messages, interaction.audit_events)

    delete_column, clear_column = st.columns(2)
    terminal = interaction.status in TERMINAL_STATUSES
    if delete_column.button(
        "Delete selected",
        key=f"delete_{interaction.id}",
        disabled=not terminal,
        width="stretch",
    ):
        delete_interaction(entry.path, entry.manifest.claim_id, interaction.id)
        st.rerun()
    has_terminal = any(item.status in TERMINAL_STATUSES for item in interactions)
    if clear_column.button(
        "Clear terminal history",
        key=f"clear_history_{entry.manifest.claim_id}",
        disabled=not has_terminal,
        width="stretch",
    ):
        clear_terminal_interactions(entry.path, entry.manifest.claim_id)
        st.rerun()


def render_native_audit(
    agent_messages: Sequence[ModelMessage],
    events: Sequence[dict[str, object]],
) -> None:
    with st.expander(
        f"Audit trail ({len(agent_messages)} messages · {len(events)} events)"
    ):
        if not agent_messages and not events:
            st.caption("No agent activity was captured before the run stopped.")
            return
        for index, message in enumerate(agent_messages, start=1):
            st.markdown(f"**Message {index} · `{message.__class__.__name__}`**")
            st.json(to_jsonable_python(message))
        if events:
            st.markdown("**Streamed events**")
            for event in events:
                st.json(event)


def claim_search_trace(
    messages: Sequence[ModelMessage],
) -> list[dict[str, object]]:
    searches: list[dict[str, object]] = []
    by_call_id: dict[str, dict[str, object]] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == "claim_search":
                args = part.args_as_dict()
                record = {
                    "query": str(args.get("query", "")),
                    "research_goal": str(args.get("research_goal", "")),
                    "source_refs": [],
                }
                searches.append(record)
                by_call_id[part.tool_call_id] = record
            elif isinstance(part, ToolReturnPart) and part.tool_name == "claim_search":
                record = by_call_id.get(part.tool_call_id)
                if record is not None:
                    record["source_refs"] = tool_source_refs(part.content)
    return searches


def tool_query(args: object) -> str:
    if isinstance(args, dict):
        return str(args.get("query", ""))
    return str(args)


def tool_source_refs(content: object) -> list[str]:
    if not isinstance(content, list):
        return []
    refs: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("source_ref"):
            refs.append(str(item["source_ref"]))
        elif hasattr(item, "source_ref"):
            refs.append(str(item.source_ref))
    return refs


def cited_answer_html(
    answer: str,
    source_refs: list[str],
    store: ClaimStore,
) -> str:
    citations = [
        (index, source_ref, _require_chunk_for_source(store, source_ref))
        for index, source_ref in enumerate(source_refs, start=1)
    ]
    rendered = html.escape(answer).replace("\n", "<br>")
    for index, source_ref, chunk in citations:
        document = get_document(store, chunk.document_id)
        tooltip = (
            f"{source_ref} · {document.title} · "
            f"pages {', '.join(chunk.page_ids)} · "
            f"{excerpt(chunk.text, 240)}"
        )
        marker = (
            f'<sup class="claim-citation" title="{html.escape(tooltip, quote=True)}">'
            f"[{index}]</sup>"
        )
        rendered = rendered.replace(f"[{html.escape(source_ref)}]", marker)
    return rendered


def excerpt(value: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _interaction_label(interaction: ResearchInteraction) -> str:
    question = excerpt(interaction.question, 70)
    return (
        f"{interaction.created_at.astimezone().strftime('%Y-%m-%d %H:%M')} · "
        f"{interaction.status} · {question}"
    )


def _render_cited_answer(answer: ClaimResearchOutput, store: ClaimStore) -> None:
    st.markdown(
        cited_answer_html(answer.answer, answer.source_refs, store),
        unsafe_allow_html=True,
    )


def _render_research_details(
    answer: ClaimResearchOutput,
    agent_messages: Sequence[ModelMessage],
    usage: dict[str, int],
    store: ClaimStore,
) -> None:
    if answer.source_refs:
        with st.expander(f"Sources ({len(answer.source_refs)})"):
            for index, source_ref in enumerate(answer.source_refs, start=1):
                chunk = _require_chunk_for_source(store, source_ref)
                document = get_document(store, chunk.document_id)
                st.markdown(f"**[{index}] {document.title}**")
                st.caption(f"{source_ref} · pages {', '.join(chunk.page_ids)}")
                st.write(excerpt(chunk.text, 500))
    searches = claim_search_trace(agent_messages)
    with st.expander(
        f"Agent trace ({len(searches)} claim searches · "
        f"{usage['requests']} model requests)"
    ):
        st.markdown("**Tool calls**")
        if not searches:
            st.caption("No retrieval tools were called.")
        for index, search in enumerate(searches, start=1):
            st.code(
                f'{index}. claim_search(query="{search["query"]}")',
                language=None,
            )
            st.caption(
                f"Goal: {search['research_goal']} · "
                f"returned {len(search['source_refs'])} chunks"
            )
        st.markdown("**Usage**")
        st.caption(
            f"{usage['requests']} model requests · "
            f"{usage['tool_calls']} tool calls · "
            f"{usage['input_tokens']} input tokens · "
            f"{usage['output_tokens']} output tokens"
        )


def _require_chunk_for_source(store: ClaimStore, source_ref: str) -> DocumentChunk:
    chunk = next(
        (chunk for chunk in store.chunks if chunk.source_ref == source_ref),
        None,
    )
    if chunk is None:
        raise ChunkNotFoundError(
            f"Citation source not found in claim {store.manifest.claim_id}: {source_ref}"
        )
    return chunk
