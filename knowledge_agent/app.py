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

from knowledge_agent.claims.config import ClaimSettings, load_claim_settings
from knowledge_agent.claims.dependencies import (
    live_ingestion_services,
    open_claim_store,
)
from knowledge_agent.claims.errors import ChunkNotFoundError
from knowledge_agent.claims.filesystem import claim_root, read_json, safe_claim_id
from knowledge_agent.claims.models import ClaimManifest, DocumentChunk, DocumentMetadata
from knowledge_agent.claims.pipeline import ingest_claim_folder, ingest_claim_pdf
from knowledge_agent.claims.store import ClaimStore, get_document, get_page, load_claim_store
from knowledge_agent.config import load_profile
from knowledge_agent.llm.client import open_structured_output_parser
from knowledge_agent.llm.config import LlmSettings, load_llm_settings
from knowledge_agent.agents.claim_researcher import (
    ChatMessage,
    ResearchAuditEntry,
    ResearchAuditTrail,
    ResearchAnswer,
    ResearchLlmAuditEntry,
    ResearchStep,
    ResearchToolAuditEntry,
    run_claim_research,
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
        _render_knowledge_base(store)
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


def _render_knowledge_base(store: ClaimStore) -> None:
    _render_claim_metrics(store)
    if not store.documents:
        st.info("This claim has no documents.")
        return

    _render_claim_overview(store)


def _render_claim_metrics(store: ClaimStore) -> None:
    manifest = store.manifest
    columns = st.columns(4)
    columns[0].metric("Claim", manifest.claim_id)
    columns[1].metric("Documents", len(manifest.documents))
    columns[2].metric("Chunks", manifest.chunk_count)
    columns[3].metric("Retrieval", manifest.retrieval_mode)


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
    st.dataframe(
        _document_rows(store),
        width="stretch",
        hide_index=True,
    )

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
    histories = st.session_state.setdefault("claim_chat_histories", {})
    messages = histories.setdefault(claim_id, [])

    header, audit_control, action = st.columns([3, 1, 1])
    header.subheader(f"Research {claim_id}")
    audit_enabled = audit_control.toggle(
        "Audit mode",
        key="research_audit_mode",
        help=(
            "Capture exact Research Agent prompts, parsed model results, and "
            "full retrieval results for this browser session."
        ),
    )
    if action.button("Clear chat", key=f"clear_chat_{claim_id}", width="stretch"):
        histories[claim_id] = []
        st.rerun()
    if audit_enabled:
        st.warning(
            "Audit traces can contain full claim evidence and conversation text. "
            "They remain in this session until you clear the chat."
        )

    _render_chat_history(messages, local_store, audit_enabled)

    prompt = st.chat_input("Ask a question about this claim")
    if not prompt:
        return
    history = _conversation_history(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if _is_greeting(prompt):
        greeting = "Hi! Ask me a question about the selected claim."
        messages.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)
        st.rerun()

    audit_entries: list[ResearchAuditEntry] = []
    with st.chat_message("assistant"):
        try:
            profile = load_profile()
            llm_settings = load_llm_settings(profile)
            with st.status("Researching the claim…", expanded=True) as status:
                st.write("Planning searches, gathering evidence, and checking gaps.")

                def show_step(step: ResearchStep) -> None:
                    status.write(step.message)

                with (
                    open_structured_output_parser(llm_settings) as parse_structured_output,
                    open_claim_store(entry.path, claim_settings) as store,
                ):
                    answer = run_claim_research(
                        store=store,
                        question=prompt,
                        parse_structured_output=parse_structured_output,
                        history=history,
                        on_step=show_step,
                        on_audit=audit_entries.append if audit_enabled else None,
                    )
                status.update(label="Research complete", state="complete")
            _render_cited_answer(answer, local_store)
            _render_research_details(answer, local_store)
            if audit_enabled:
                _render_audit_trail(ResearchAuditTrail(entries=audit_entries))
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            if audit_enabled:
                messages[-1]["exclude_from_history"] = True
                trail = ResearchAuditTrail(entries=audit_entries)
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Research failed: {error}",
                        "research_error": error,
                        "exclude_from_history": True,
                        "audit": trail.model_dump(mode="json"),
                    }
                )
                st.error(f"Research failed: {error}")
                _render_audit_trail(trail)
            else:
                st.error(error)
            return

    assistant_message = {
        "role": "assistant",
        "content": answer.answer,
        "research": answer.model_dump(mode="json"),
    }
    if audit_enabled:
        assistant_message["audit"] = ResearchAuditTrail(
            entries=audit_entries
        ).model_dump(mode="json")
    messages.append(assistant_message)
    st.rerun()


def _conversation_history(
    messages: Sequence[dict[str, object]],
) -> list[ChatMessage]:
    return [
        ChatMessage(role=message["role"], content=message["content"])
        for message in messages
        if not message.get("exclude_from_history", False)
    ]


def _render_chat_history(
    messages: Sequence[dict[str, object]],
    store: ClaimStore,
    audit_enabled: bool,
) -> None:
    for message in messages:
        role = str(message["role"])
        with st.chat_message(role):
            if message["role"] == "assistant" and "research_error" in message:
                st.error(f"Research failed: {message['research_error']}")
                _render_saved_audit(message, audit_enabled)
            elif message["role"] == "assistant" and "research" in message:
                try:
                    answer = ResearchAnswer.model_validate(message["research"])
                    _render_cited_answer(answer, store)
                    _render_research_details(answer, store)
                except Exception as exc:
                    st.error(f"Could not display saved research answer: {exc}")
                _render_saved_audit(message, audit_enabled)
            else:
                st.markdown(str(message["content"]))


def _render_saved_audit(
    message: dict[str, object],
    audit_enabled: bool,
) -> None:
    if not audit_enabled:
        return
    saved_audit = message.get("audit")
    if saved_audit is None:
        st.caption("Audit mode was not enabled for this turn.")
        return
    try:
        trail = ResearchAuditTrail.model_validate(saved_audit)
    except Exception as exc:
        st.error(f"Could not display saved audit trail: {exc}")
        return
    _render_audit_trail(trail)


def _render_audit_trail(trail: ResearchAuditTrail) -> None:
    entry_count = len(trail.entries)
    entry_label = "entry" if entry_count == 1 else "entries"
    with st.expander(f"Audit trail ({entry_count} {entry_label})"):
        if not trail.entries:
            st.caption("No agent calls were captured before the run stopped.")
            return
        for index, entry in enumerate(trail.entries, start=1):
            with st.container(border=True):
                if isinstance(entry, ResearchLlmAuditEntry):
                    st.markdown(
                        f"**{index}. LLM · `{entry.operation}` "
                        f"→ `{entry.response_model}`**"
                    )
                    system_tab, user_tab, result_tab = st.tabs(
                        ["System prompt", "User prompt", "Result"]
                    )
                    with system_tab:
                        st.code(entry.system_prompt, language=None, wrap_lines=True)
                    with user_tab:
                        st.code(entry.user_prompt, language=None, wrap_lines=True)
                    with result_tab:
                        if entry.error is not None:
                            st.error(entry.error)
                        else:
                            st.json(entry.result)
                elif isinstance(entry, ResearchToolAuditEntry):
                    st.markdown(f"**{index}. Tool · `{entry.tool_name}`**")
                    input_tab, result_tab = st.tabs(["Input", "Result"])
                    with input_tab:
                        st.json(
                            {
                                "query": entry.query.query,
                                "research_goal": entry.query.research_goal,
                                "top_k": entry.top_k,
                            }
                        )
                    with result_tab:
                        if entry.error is not None:
                            st.error(entry.error)
                        else:
                            st.json(
                                [
                                    item.model_dump(mode="json")
                                    for item in entry.result or []
                                ]
                            )


def _render_cited_answer(answer: ResearchAnswer, store: ClaimStore) -> None:
    st.markdown(
        _cited_answer_html(answer.answer, answer.source_refs, store),
        unsafe_allow_html=True,
    )


def _render_research_details(answer: ResearchAnswer, store: ClaimStore) -> None:
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
    with st.expander(
        f"Agent steps ({len(answer.steps)}) · tool calls ({len(answer.searches)})"
    ):
        st.markdown("**Execution steps**")
        for index, step in enumerate(answer.steps, start=1):
            st.markdown(f"{index}. `{step.stage}` — {step.message}")

        st.markdown("**Tool calls**")
        if not answer.searches:
            st.caption("No retrieval tools were called.")
        for index, search in enumerate(answer.searches, start=1):
            st.code(
                f'{index}. claim_search(query="{search.query.query}")',
                language=None,
            )
            st.caption(
                f"Goal: {search.query.research_goal} · "
                f"returned {len(search.source_refs)} chunks"
            )

        if answer.gap_reviews:
            st.markdown("**Gap reviews**")
            for review in answer.gap_reviews:
                label = "complete" if review.complete else "more research needed"
                st.markdown(
                    f"- **{label}:** "
                    f"{len(review.missing_information)} gaps, "
                    f"{len(review.queries)} follow-up queries"
                )

        st.markdown("**Objectives**")
        for objective in answer.plan.objectives:
            st.markdown(f"- {objective}")
        st.markdown("**Searches**")
        for search in answer.searches:
            st.markdown(
                f"- `{search.query.query}` — {search.query.research_goal} "
                f"({len(search.source_refs)} hits)"
            )
        st.markdown("**Validated findings**")
        if not answer.findings:
            st.caption("No supported findings were extracted.")
        for finding in answer.findings:
            st.markdown(f"- {finding.insight}")


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
