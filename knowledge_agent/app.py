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

from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import (
    live_ingestion_services,
    open_claim_store,
)
from knowledge_agent.claims.filesystem import claim_root, read_json, safe_claim_id
from knowledge_agent.claims.models import ClaimManifest, DocumentChunk
from knowledge_agent.claims.pipeline import ingest_claim_folder, ingest_claim_pdf
from knowledge_agent.claims.store import ClaimStore
from knowledge_agent.config import load_profile
from knowledge_agent.llm.client import open_responses_client
from knowledge_agent.llm.config import LlmSettings
from knowledge_agent.research.agent import run_claim_research
from knowledge_agent.research.llm import ResponsesResearchModel
from knowledge_agent.research.models import ChatMessage, ResearchAnswer, ResearchStep


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

    claim_settings = ClaimSettings.from_env()
    claims, invalid_claims = discover_claims(claim_settings.data_root)
    selected = _render_sidebar(claim_settings, claims, invalid_claims)
    notice = st.session_state.pop("claim_notice", None)
    if notice:
        st.success(notice)

    if selected is None:
        st.info("Ingest a claim from the sidebar to begin.")
        return

    try:
        store = ClaimStore(selected.path)
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
        llm_settings = LlmSettings.from_env(profile)
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
    manifest = store.manifest
    columns = st.columns(4)
    columns[0].metric("Claim", manifest.claim_id)
    columns[1].metric("Documents", len(manifest.documents))
    columns[2].metric("Chunks", manifest.chunk_count)
    columns[3].metric("Retrieval", manifest.retrieval_mode)

    if not store.documents:
        st.info("This claim has no documents.")
        return

    st.subheader("Claim overview")
    timeline_tab, parties_tab = st.tabs(["Timeline", "Parties"])
    with timeline_tab:
        _render_timeline(store)
    with parties_tab:
        parties = _party_rows(store)
        if parties:
            st.dataframe(parties, width="stretch", hide_index=True)
        else:
            st.caption("No parties were extracted from this claim.")

    st.subheader("Documents")
    st.dataframe(
        [
            {
                "ID": document.id,
                "Type": document.document_type,
                "Title": document.title,
                "Pages": (
                    f"{document.page_range.start_page}–{document.page_range.end_page}"
                ),
                "File": document.file_name,
                "Summary": document.summary,
            }
            for document in store.documents
        ],
        width="stretch",
        hide_index=True,
    )
    selected_document_id = st.selectbox(
        "Inspect document",
        [document.id for document in store.documents],
        format_func=lambda document_id: (
            f"{document_id} — {store.get_document(document_id).title}"
        ),
    )
    document = store.get_document(selected_document_id)
    st.markdown(f"**{document.title}**  \n{document.summary}")

    parties_column, events_column = st.columns(2)
    with parties_column:
        st.markdown("#### Involved parties")
        if document.involved_parties:
            st.dataframe(
                [party.model_dump() for party in document.involved_parties],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No parties extracted.")
    with events_column:
        st.markdown("#### Events")
        if document.events:
            st.dataframe(
                [event.model_dump() for event in document.events],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No events extracted.")

    st.subheader("Evidence chunks")
    filter_text = st.text_input("Filter chunks", placeholder="Search text or source reference")
    document_chunks = [
        chunk for chunk in store.chunks if chunk.document_id == selected_document_id
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
            "Chunk text",
            [chunk.source_ref for chunk in document_chunks],
        )
        selected_chunk = _chunk_by_ref(document_chunks, selected_ref)
        st.code(selected_chunk.text, language=None, wrap_lines=True)

    st.subheader("OCR pages")
    pages = [
        page
        for page in store.pages
        if document.page_range.start_page
        <= page.page_number
        <= document.page_range.end_page
    ]
    selected_page_id = st.selectbox("Page text", [page.page_id for page in pages])
    st.code(store.get_page(selected_page_id).text, language=None, wrap_lines=True)


def _render_chat(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    local_store: ClaimStore,
) -> None:
    claim_id = entry.manifest.claim_id
    histories = st.session_state.setdefault("claim_chat_histories", {})
    messages = histories.setdefault(claim_id, [])

    header, action = st.columns([4, 1])
    header.subheader(f"Research {claim_id}")
    if action.button("Clear chat", key=f"clear_chat_{claim_id}", width="stretch"):
        histories[claim_id] = []
        st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "research" in message:
                answer = ResearchAnswer.model_validate(message["research"])
                _render_cited_answer(answer, local_store)
                _render_research_details(answer, local_store)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about this claim")
    if not prompt:
        return
    history = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in messages
    ]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if _is_greeting(prompt):
        greeting = "Hi! Ask me a question about the selected claim."
        messages.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)
        st.rerun()

    try:
        profile = load_profile()
        llm_settings = LlmSettings.from_env(profile)
        with st.chat_message("assistant"):
            with st.status("Researching the claim…", expanded=True) as status:
                st.write("Planning searches, gathering evidence, and checking gaps.")

                def show_step(step: ResearchStep) -> None:
                    status.write(step.message)

                with (
                    open_responses_client(llm_settings) as responses,
                    open_claim_store(entry.path, claim_settings) as store,
                ):
                    answer = run_claim_research(
                        store=store,
                        question=prompt,
                        model=ResponsesResearchModel(responses),
                        history=history,
                        on_step=show_step,
                    )
                status.update(label="Research complete", state="complete")
            _render_cited_answer(answer, local_store)
            _render_research_details(answer, local_store)
    except Exception as exc:
        st.error(str(exc))
        return

    messages.append(
        {
            "role": "assistant",
            "content": answer.answer,
            "research": answer.model_dump(mode="json"),
        }
    )
    st.rerun()


def _render_cited_answer(answer: ResearchAnswer, store: ClaimStore) -> None:
    st.markdown(
        _cited_answer_html(answer.answer, answer.source_refs, store),
        unsafe_allow_html=True,
    )


def _render_research_details(answer: ResearchAnswer, store: ClaimStore) -> None:
    if answer.source_refs:
        with st.expander(f"Sources ({len(answer.source_refs)})"):
            for index, source_ref in enumerate(answer.source_refs, start=1):
                chunk = _chunk_for_source(store, source_ref)
                if chunk is None:
                    st.code(f"[{index}] {source_ref}", language=None)
                    continue
                document = store.get_document(chunk.document_id)
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
    rendered = html.escape(answer).replace("\n", "<br>")
    for index, source_ref in enumerate(source_refs, start=1):
        chunk = _chunk_for_source(store, source_ref)
        if chunk is None:
            tooltip = source_ref
        else:
            document = store.get_document(chunk.document_id)
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


def _chunk_for_source(store: ClaimStore, source_ref: str) -> DocumentChunk | None:
    return next(
        (chunk for chunk in store.chunks if chunk.source_ref == source_ref),
        None,
    )


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
