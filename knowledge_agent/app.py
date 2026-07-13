"""Streamlit workbench for claim ingestion, inspection, and research chat."""

from __future__ import annotations

import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, Sequence

import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import (
    ModelMessage,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_core import to_jsonable_python

from knowledge_agent.claims.config import ClaimSettings, load_claim_settings
from knowledge_agent.claims.dependencies import (
    live_ingestion_services,
    open_claim_store,
    rebuild_claim_knowledge_base,
)
from knowledge_agent.claims.errors import ChunkNotFoundError
from knowledge_agent.claims.filesystem import claim_root, read_json, safe_claim_id
from knowledge_agent.claims.lightrag import (
    load_lightrag_graph,
    load_lightrag_metadata,
)
from knowledge_agent.claims.models import (
    ClaimManifest,
    DocumentChunk,
    DocumentMetadata,
    KnowledgeBaseEngine,
)
from knowledge_agent.claims.pipeline import ingest_claim_folder, ingest_claim_pdf
from knowledge_agent.claims.store import ClaimStore, get_document, get_page, load_claim_store
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import LlmSettings, load_llm_settings
from knowledge_agent.llm.providers import open_agent_runtime
from knowledge_agent.agents.claim_researcher import (
    ClaimResearchPlan,
    ClaimResearchOutput,
    ResearchClarification,
    run_claim_planning,
    run_claim_research,
)
from knowledge_agent.research.history import (
    TERMINAL_STATUSES,
    ClarificationExchange,
    InteractionUsage,
    KnowledgeBaseSnapshot,
    ResearchHistory,
    ResearchInteraction,
    clear_terminal_interactions,
    delete_interaction,
    load_research_history,
    store_interaction,
    utc_now,
)


UploadMode = Literal["combined", "separate"]
COMBINED_LABEL = "One combined claim PDF"
SEPARATE_LABEL = "Separate document PDFs"


class UploadedPdf(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


@dataclass(frozen=True)
class ClaimEntry:
    path: Path
    manifest: ClaimManifest


@dataclass(frozen=True)
class InvalidClaim:
    path: Path
    error: str


def discover_claims(data_root: Path) -> tuple[list[ClaimEntry], list[InvalidClaim]]:
    if not data_root.exists():
        return [], []
    claims: list[ClaimEntry] = []
    invalid: list[InvalidClaim] = []
    for path in sorted(
        (item for item in data_root.iterdir() if item.is_dir()),
        key=lambda item: item.name.casefold(),
    ):
        try:
            manifest_data = read_json(path / "manifest.json")
            if not isinstance(manifest_data, dict):
                raise ValueError("manifest.json must contain a JSON object")
            manifest = ClaimManifest.model_validate(manifest_data)
        except Exception as exc:
            invalid.append(InvalidClaim(path=path, error=str(exc)))
        else:
            claims.append(ClaimEntry(path=path, manifest=manifest))
    return claims, invalid


def validate_uploads(mode: UploadMode, uploads: Sequence[UploadedPdf]) -> None:
    if not uploads:
        raise ValueError("Select at least one PDF")
    if mode == "combined" and len(uploads) != 1:
        raise ValueError("Combined claim mode requires exactly one PDF")
    names: set[str] = set()
    for upload in uploads:
        name = Path(upload.name).name
        if not name or Path(name).suffix.casefold() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {upload.name}")
        key = name.casefold()
        if key in names:
            raise ValueError(f"Duplicate uploaded file name: {name}")
        names.add(key)


def ingest_uploads(
    claim_id: str,
    mode: UploadMode,
    uploads: Sequence[UploadedPdf],
    claim_settings: ClaimSettings,
    llm_settings: LlmSettings,
    knowledge_base: KnowledgeBaseEngine = "custom",
) -> ClaimManifest:
    claim_id = safe_claim_id(claim_id)
    validate_uploads(mode, uploads)
    output_path = claim_root(claim_settings.data_root, claim_id)
    if output_path.exists():
        raise FileExistsError(
            f"Claim {claim_id!r} already exists at {output_path}. "
            "Choose a different claim ID."
        )

    try:
        with TemporaryDirectory(prefix="knowledge-agent-upload-") as temporary_dir:
            upload_root = Path(temporary_dir)
            paths = []
            for upload in uploads:
                path = upload_root / Path(upload.name).name
                path.write_bytes(upload.getvalue())
                paths.append(path)
            with live_ingestion_services(
                claim_id,
                claim_settings,
                llm_settings,
                knowledge_base,
            ) as services:
                if mode == "combined":
                    return ingest_claim_pdf(
                        claim_id,
                        paths[0],
                        claim_settings.data_root,
                        services,
                    )
                return ingest_claim_folder(
                    claim_id,
                    upload_root,
                    claim_settings.data_root,
                    services,
                )
    except Exception:
        if output_path.exists():
            shutil.rmtree(output_path)
        raise


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Claim Research Workbench", layout="wide")
    st.markdown(
        """
        <style>
        .claim-citation {
            color: #2563eb;
            cursor: help;
            font-size: 0.78em;
            font-weight: 700;
            padding-left: 0.1rem;
        }
        .claim-timeline {
            border-left: 2px solid #cbd5e1;
            margin: 0.5rem 0 1rem 0.5rem;
            padding-left: 1.25rem;
        }
        .claim-timeline-item {
            margin: 0 0 1.2rem 0;
            position: relative;
        }
        .claim-timeline-item::before {
            background: #2563eb;
            border-radius: 50%;
            content: "";
            height: 0.65rem;
            left: -1.63rem;
            position: absolute;
            top: 0.35rem;
            width: 0.65rem;
        }
        .claim-timeline-date { font-weight: 700; }
        .claim-timeline-source { color: #64748b; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Claim Research Workbench")
    st.caption("Ingest claim PDFs, inspect their evidence, and ask cited questions.")

    claim_settings = load_claim_settings()
    claims, invalid_claims = discover_claims(claim_settings.data_root)
    selected = _render_sidebar(claim_settings, claims, invalid_claims)
    notice = st.session_state.pop("claim_notice", None)
    if notice:
        st.success(notice)

    if selected is None:
        st.info("Ingest a claim from the sidebar to begin.")
        return

    try:
        store = load_claim_store(selected.path)
    except Exception as exc:
        st.error(f"Could not open claim {selected.manifest.claim_id}: {exc}")
        return

    knowledge_tab, chat_tab = st.tabs(["Knowledge base", "Research chat"])
    with knowledge_tab:
        _render_knowledge_base(selected, claim_settings, store)
    with chat_tab:
        _render_chat(selected, claim_settings, store)


def _render_sidebar(
    claim_settings: ClaimSettings,
    claims: list[ClaimEntry],
    invalid_claims: list[InvalidClaim],
) -> ClaimEntry | None:
    with st.sidebar:
        st.header("Claims")
        by_id = {entry.manifest.claim_id: entry for entry in claims}
        claim_ids = list(by_id)
        preferred = st.session_state.get("selected_claim_id")
        index = (
            claim_ids.index(preferred)
            if preferred in claim_ids
            else (0 if claim_ids else None)
        )
        selected_id = st.selectbox(
            "Active claim",
            claim_ids,
            index=index,
            placeholder="No valid claims found",
            disabled=not claim_ids,
        )
        if selected_id:
            st.session_state.selected_claim_id = selected_id

        with st.expander("Ingest a new claim", expanded=not claims):
            _render_ingestion_form(claim_settings)

        if invalid_claims:
            with st.expander(f"Unavailable claim folders ({len(invalid_claims)})"):
                for invalid in invalid_claims:
                    st.error(f"{invalid.path.name}: {invalid.error}")
        st.caption(f"Data root: {claim_settings.data_root}")
    return by_id.get(selected_id) if selected_id else None


def _render_ingestion_form(claim_settings: ClaimSettings) -> None:
    claim_id = st.text_input("Claim ID", placeholder="CLM-001")
    knowledge_base_label = st.radio(
        "Knowledge base",
        ["Custom", "LightRAG"],
        help=(
            "Custom uses the current lexical or Snowflake/Chroma index. "
            "LightRAG builds a claim-local entity graph and vector index."
        ),
    )
    knowledge_base: KnowledgeBaseEngine = (
        "lightrag" if knowledge_base_label == "LightRAG" else "custom"
    )
    mode_label = st.radio(
        "PDF layout",
        [SEPARATE_LABEL, COMBINED_LABEL],
        help=(
            "Use separate PDFs when each upload is already one logical document. "
            "Use combined mode when one PDF must be split into documents."
        ),
    )
    mode: UploadMode = "combined" if mode_label == COMBINED_LABEL else "separate"
    uploads = st.file_uploader(
        "Claim PDFs",
        type=["pdf"],
        accept_multiple_files=mode == "separate",
        key=f"claim_uploads_{mode}",
    )
    upload_list = list(uploads or []) if mode == "separate" else ([uploads] if uploads else [])
    if not st.button("Ingest claim", type="primary", width="stretch"):
        return

    try:
        profile = load_profile()
        llm_settings = load_llm_settings(profile)
        validate_uploads(mode, upload_list)
        with st.status("Ingesting claim…", expanded=True) as status:
            st.write("Running OCR, document classification, chunking, and indexing.")
            manifest = ingest_uploads(
                claim_id,
                mode,
                upload_list,
                claim_settings,
                llm_settings,
                knowledge_base,
            )
            status.update(label="Claim ingested", state="complete")
    except Exception as exc:
        st.error(f"{exc} No partial claim was retained; you can retry this ID.")
        return

    st.session_state.selected_claim_id = manifest.claim_id
    st.session_state.claim_notice = (
        f"Ingested {manifest.claim_id}: {len(manifest.documents)} documents, "
        f"{manifest.chunk_count} chunks."
    )
    st.rerun()


def _render_knowledge_base(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    store: ClaimStore,
) -> None:
    _render_claim_metrics(store)
    _render_rebuild_knowledge_base(entry, claim_settings, store)
    if not store.documents:
        st.info("This claim has no documents.")
        return

    _render_claim_overview(store)
    if store.manifest.retrieval_mode == "lightrag":
        _render_lightrag_graph(entry.path)


def _render_claim_metrics(store: ClaimStore) -> None:
    manifest = store.manifest
    columns = st.columns(4)
    columns[0].metric("Claim", manifest.claim_id)
    columns[1].metric("Documents", len(manifest.documents))
    columns[2].metric("Chunks", manifest.chunk_count)
    columns[3].metric("Retrieval", manifest.retrieval_mode)


def _render_rebuild_knowledge_base(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    store: ClaimStore,
) -> None:
    try:
        history = load_research_history(entry.path, entry.manifest.claim_id)
        active = history.active_interaction()
        history_error = None
    except Exception as exc:
        active = None
        history_error = str(exc)

    with st.expander("Rebuild knowledge base"):
        st.caption(
            "Rebuild from saved chunks without rerunning OCR or metadata extraction. "
            "Saved research reports are not changed."
        )
        if history_error:
            st.error(f"Could not validate active research: {history_error}")
        if active is not None:
            st.warning(
                "Finish or cancel the active research interaction before rebuilding."
            )
        target_label = st.selectbox(
            "Target engine",
            ["Custom", "LightRAG"],
            index=1 if store.manifest.retrieval_mode == "lightrag" else 0,
            key=f"rebuild_engine_{entry.manifest.claim_id}",
        )
        disabled = active is not None or history_error is not None
        if not st.button(
            "Rebuild index",
            key=f"rebuild_index_{entry.manifest.claim_id}",
            disabled=disabled,
            width="stretch",
        ):
            return
        target: KnowledgeBaseEngine = (
            "lightrag" if target_label == "LightRAG" else "custom"
        )
        try:
            llm_settings = load_llm_settings(load_profile())
            with st.status("Rebuilding knowledge base…", expanded=True) as status:
                status.write("Indexing the persisted claim chunks.")
                manifest = rebuild_claim_knowledge_base(
                    entry.path,
                    target,
                    claim_settings,
                    llm_settings,
                )
                status.update(label="Knowledge base rebuilt", state="complete")
        except Exception as exc:
            st.error(f"Knowledge base rebuild failed: {exc}")
            return
        st.session_state.claim_notice = (
            f"Rebuilt {manifest.claim_id} with {manifest.retrieval_mode} retrieval."
        )
        st.rerun()


def _render_lightrag_graph(claim_path: Path) -> None:
    st.subheader("LightRAG graph")
    try:
        index_path = claim_path / "index" / "lightrag"
        metadata = load_lightrag_metadata(index_path)
        graph = load_lightrag_graph(index_path)
    except Exception as exc:
        st.error(f"Could not load the LightRAG graph: {exc}")
        return

    columns = st.columns(3)
    columns[0].metric("Entities", metadata.entity_count)
    columns[1].metric("Relationships", metadata.relationship_count)
    columns[2].metric("Indexed chunks", metadata.indexed_chunk_count)
    st.caption(
        f"LightRAG {metadata.lightrag_version} · {metadata.llm_model} · "
        f"{metadata.embedding_model}"
    )
    entity_tab, relationship_tab = st.tabs(["Entities", "Relationships"])
    with entity_tab:
        entity_filter = st.text_input(
            "Filter entities",
            key=f"entity_filter_{metadata.claim_id}",
        )
        _render_graph_table(graph.entities, entity_filter, "entities")
    with relationship_tab:
        relationship_filter = st.text_input(
            "Filter relationships",
            key=f"relationship_filter_{metadata.claim_id}",
        )
        _render_graph_table(
            graph.relationships,
            relationship_filter,
            "relationships",
        )


def _render_graph_table(
    rows: list[dict[str, object]],
    filter_text: str,
    label: str,
) -> None:
    needle = filter_text.strip().casefold()
    filtered = [
        row
        for row in rows
        if not needle
        or needle in " ".join(str(value) for value in row.values()).casefold()
    ]
    displayed = filtered[:500]
    if displayed:
        st.dataframe(displayed, width="stretch", hide_index=True)
    else:
        st.caption(f"No {label} match this filter.")
    if len(filtered) > len(displayed):
        st.info(f"Showing 500 of {len(filtered)} matching {label}.")


def _render_claim_overview(store: ClaimStore) -> None:
    st.subheader("Claim overview")
    timeline_tab, parties_tab, documents_tab = st.tabs(
        ["Timeline", "Parties", "Documents"]
    )
    with timeline_tab:
        _render_timeline(store)
    with parties_tab:
        parties = _party_rows(store)
        if parties:
            st.dataframe(parties, width="stretch", hide_index=True)
        else:
            st.caption("No parties were extracted from this claim.")
    with documents_tab:
        _render_documents(store)


def _render_documents(store: ClaimStore) -> None:
    st.markdown("#### Document inventory")
    st.caption(
        "Browse the claim file, compare extracted metadata, and inspect the "
        "evidence behind each document."
    )
    st.dataframe(_document_rows(store), width="stretch", hide_index=True)

    document = _select_document(store)
    document_chunks = [
        chunk for chunk in store.chunks if chunk.document_id == document.id
    ]
    page_range = _page_range_label(document)
    columns = st.columns(3)
    columns[0].metric("Selected document", document.id)
    columns[1].metric("Pages", page_range)
    columns[2].metric("Evidence chunks", len(document_chunks))

    st.markdown(f"### {document.title}")
    st.caption(
        f"{document.document_type} · {document.file_name} · pages {page_range}"
    )
    metadata_tab, evidence_tab, pages_tab = st.tabs(
        ["Metadata", "Evidence", "OCR pages"]
    )
    with metadata_tab:
        _render_document_metadata(document)
    with evidence_tab:
        _render_document_chunks(store, document.id)
    with pages_tab:
        _render_document_pages(store, document)


def _document_rows(store: ClaimStore) -> list[dict[str, str | int]]:
    chunk_counts: dict[str, int] = {}
    for chunk in store.chunks:
        chunk_counts[chunk.document_id] = chunk_counts.get(chunk.document_id, 0) + 1
    return [
        {
            "ID": document.id,
            "Type": document.document_type,
            "Title": document.title,
            "Pages": _page_range_label(document),
            "Parties": len(document.involved_parties),
            "Events": len(document.events),
            "Evidence chunks": chunk_counts.get(document.id, 0),
            "File": document.file_name,
        }
        for document in store.documents
    ]


def _page_range_label(document: DocumentMetadata) -> str:
    start = document.page_range.start_page
    end = document.page_range.end_page
    return str(start) if start == end else f"{start}–{end}"


def _select_document(store: ClaimStore) -> DocumentMetadata:
    selected_document_id = st.selectbox(
        "Select a document to inspect",
        [document.id for document in store.documents],
        format_func=lambda document_id: (
            f"{document_id} — {get_document(store, document_id).title}"
        ),
        key=f"document_select_{store.manifest.claim_id}",
    )
    return get_document(store, selected_document_id)


def _render_document_metadata(document: DocumentMetadata) -> None:
    with st.container(border=True):
        st.markdown("##### Summary")
        st.write(document.summary)

    parties_column, events_column = st.columns(2)
    with parties_column:
        st.markdown("##### Involved parties")
        if document.involved_parties:
            st.dataframe(
                [party.model_dump() for party in document.involved_parties],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No parties extracted.")
    with events_column:
        st.markdown("##### Events")
        if document.events:
            st.dataframe(
                [event.model_dump() for event in document.events],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No events extracted.")


def _render_document_chunks(store: ClaimStore, document_id: str) -> None:
    st.markdown("##### Evidence chunks")
    filter_text = st.text_input(
        "Filter chunks",
        placeholder="Search text or source reference",
        key=f"chunk_filter_{store.manifest.claim_id}_{document_id}",
    )
    document_chunks = [
        chunk for chunk in store.chunks if chunk.document_id == document_id
    ]
    if filter_text.strip():
        needle = filter_text.casefold()
        document_chunks = [
            chunk
            for chunk in document_chunks
            if needle in chunk.text.casefold() or needle in chunk.source_ref.casefold()
        ]
    if not document_chunks:
        st.info("No chunks match this filter.")
    else:
        st.dataframe(
            [
                {
                    "Source ref": chunk.source_ref,
                    "Pages": ", ".join(chunk.page_ids),
                    "Text": chunk.text,
                }
                for chunk in document_chunks
            ],
            width="stretch",
            hide_index=True,
        )
        selected_ref = st.selectbox(
            "Inspect chunk",
            [chunk.source_ref for chunk in document_chunks],
            key=f"chunk_select_{store.manifest.claim_id}_{document_id}",
        )
        selected_chunk = _chunk_by_ref(document_chunks, selected_ref)
        st.code(selected_chunk.text, language=None, wrap_lines=True)


def _render_document_pages(
    store: ClaimStore,
    document: DocumentMetadata,
) -> None:
    st.markdown("##### OCR pages")
    pages = [
        page
        for page in store.pages
        if document.page_range.start_page
        <= page.page_number
        <= document.page_range.end_page
    ]
    selected_page_id = st.selectbox(
        "Inspect page",
        [page.page_id for page in pages],
        key=f"page_select_{store.manifest.claim_id}_{document.id}",
    )
    st.code(get_page(store, selected_page_id).text, language=None, wrap_lines=True)


def _render_chat(
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
        _render_report_history(entry, local_store, history)


def _render_new_research(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    store: ClaimStore,
    history: ResearchHistory,
) -> None:
    claim_id = entry.manifest.claim_id
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

    active = history.active_interaction()
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
    if _is_greeting(prompt):
        st.info("Ask a specific question about the selected claim.")
        return

    try:
        knowledge_base = _knowledge_base_snapshot(entry)
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
    _render_knowledge_base_snapshot(interaction.knowledge_base)
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
        _render_plan(interaction.plan)
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
                _run_planning_phase(entry, store, interaction, prompt or interaction.question)
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
        _render_native_audit(interaction.agent_messages, interaction.audit_events)


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
        settings = load_llm_settings(load_profile())
        with st.status("Researching the claim…", expanded=True) as status:
            status.write("Searching the selected claim and preparing the report.")
            with (
                open_agent_runtime(settings) as runtime,
                open_claim_store(
                    entry.path,
                    claim_settings,
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
            status.write(f"Searched claim evidence: {_tool_query(part.get('args'))}")
        elif event.get("type") == "FunctionToolResultEvent":
            status.write(
                "Claim search returned "
                f"{len(_tool_source_refs(part.get('content')))} chunks."
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


def _knowledge_base_snapshot(entry: ClaimEntry) -> KnowledgeBaseSnapshot:
    manifest = entry.manifest
    if manifest.retrieval_mode != "lightrag":
        return KnowledgeBaseSnapshot(
            retrieval_mode=manifest.retrieval_mode,
            embedding_provider=manifest.embedding_provider,
            embedding_model=manifest.embedding_model,
        )
    metadata = load_lightrag_metadata(entry.path / "index" / "lightrag")
    if metadata.claim_id != manifest.claim_id:
        raise ValueError("LightRAG metadata claim_id does not match the claim")
    return KnowledgeBaseSnapshot(
        retrieval_mode=manifest.retrieval_mode,
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


def _render_knowledge_base_snapshot(
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


def _render_plan(plan: ClaimResearchPlan | None) -> None:
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


def _render_report_history(
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
    _render_knowledge_base_snapshot(interaction.knowledge_base)
    if interaction.clarifications:
        with st.expander(f"Clarifications ({len(interaction.clarifications)})"):
            for exchange in interaction.clarifications:
                st.markdown(f"**Agent:** {exchange.question}")
                st.markdown(f"**User:** {exchange.answer or '(unanswered)'}")
    if interaction.plan is not None:
        st.markdown("### Approved plan")
        _render_plan(interaction.plan)
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
    _render_native_audit(interaction.agent_messages, interaction.audit_events)

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


def _interaction_label(interaction: ResearchInteraction) -> str:
    question = _excerpt(interaction.question, 70)
    return (
        f"{interaction.created_at.astimezone().strftime('%Y-%m-%d %H:%M')} · "
        f"{interaction.status} · {question}"
    )


def _render_native_audit(
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


def _render_cited_answer(answer: ClaimResearchOutput, store: ClaimStore) -> None:
    st.markdown(
        _cited_answer_html(answer.answer, answer.source_refs, store),
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
                st.caption(
                    f"{source_ref} · pages {', '.join(chunk.page_ids)}"
                )
                st.write(_excerpt(chunk.text, 500))
    searches = _claim_search_trace(agent_messages)
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
            f"{usage['requests']} model requests · {usage['tool_calls']} tool calls · "
            f"{usage['input_tokens']} input tokens · "
            f"{usage['output_tokens']} output tokens"
        )


def _claim_search_trace(
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
                    record["source_refs"] = _tool_source_refs(part.content)
    return searches


def _tool_query(args: object) -> str:
    if isinstance(args, dict):
        return str(args.get("query", ""))
    return str(args)


def _tool_source_refs(content: object) -> list[str]:
    if not isinstance(content, list):
        return []
    refs: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("source_ref"):
            refs.append(str(item["source_ref"]))
        elif hasattr(item, "source_ref"):
            refs.append(str(item.source_ref))
    return refs


def _usage_payload(usage: object) -> dict[str, int]:
    return {
        "requests": int(getattr(usage, "requests", 0)),
        "tool_calls": int(getattr(usage, "tool_calls", 0)),
        "input_tokens": int(getattr(usage, "input_tokens", 0)),
        "output_tokens": int(getattr(usage, "output_tokens", 0)),
    }


def _render_timeline(store: ClaimStore) -> None:
    events = _timeline_rows(store)
    if not events:
        st.caption("No events were extracted from this claim.")
        return
    items = []
    for event in events:
        items.append(
            '<div class="claim-timeline-item">'
            f'<div class="claim-timeline-date">{html.escape(event["Date"])}</div>'
            f'<div>{html.escape(event["Event"])}</div>'
            '<div class="claim-timeline-source" '
            f'title="{html.escape(event["Source ref"], quote=True)}">'
            f'{html.escape(event["Document"])}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="claim-timeline">' + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )


def _timeline_rows(store: ClaimStore) -> list[dict[str, str]]:
    rows: list[tuple[tuple[object, ...], dict[str, str]]] = []
    seen: set[tuple[int | None, int | None, int | None, str, str]] = set()
    for document in store.documents:
        for event in document.events:
            key = (
                event.year,
                event.month,
                event.day,
                event.sentence,
                event.source_ref,
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                (
                    (
                        event.year is None,
                        event.year or 9999,
                        event.month or 13,
                        event.day or 32,
                        event.sentence.casefold(),
                    ),
                    {
                    "Date": _event_date(event.year, event.month, event.day),
                    "Event": event.sentence,
                    "Document": document.title,
                    "Source ref": event.source_ref,
                    },
                )
            )
    return [row for _, row in sorted(rows, key=lambda item: item[0])]


def _party_rows(store: ClaimStore) -> list[dict[str, str]]:
    parties: dict[str, tuple[str, set[str], set[str]]] = {}
    for document in store.documents:
        for party in document.involved_parties:
            key = party.name.casefold()
            name, roles, documents = parties.setdefault(
                key,
                (party.name, set(), set()),
            )
            roles.add(party.role)
            documents.add(document.title)
    return [
        {
            "Party": name,
            "Roles": ", ".join(sorted(roles, key=str.casefold)),
            "Documents": ", ".join(sorted(documents, key=str.casefold)),
        }
        for name, roles, documents in sorted(
            parties.values(),
            key=lambda entry: entry[0].casefold(),
        )
    ]


def _cited_answer_html(
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
            f"{_excerpt(chunk.text, 240)}"
        )
        marker = (
            f'<sup class="claim-citation" title="{html.escape(tooltip, quote=True)}">'
            f"[{index}]</sup>"
        )
        rendered = rendered.replace(
            f"[{html.escape(source_ref)}]",
            marker,
        )
    return rendered


def _event_date(
    year: int | None,
    month: int | None,
    day: int | None,
) -> str:
    if year is None:
        return "Undated"
    if month is None:
        return str(year)
    if day is None:
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}-{month:02d}-{day:02d}"


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


def _excerpt(value: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _is_greeting(value: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", value.casefold()).strip()
    return normalized in {"hi", "hello", "hey", "good morning", "good evening"}


def _chunk_by_ref(chunks: list[DocumentChunk], source_ref: str) -> DocumentChunk:
    return next(chunk for chunk in chunks if chunk.source_ref == source_ref)


if __name__ == "__main__":
    main()
