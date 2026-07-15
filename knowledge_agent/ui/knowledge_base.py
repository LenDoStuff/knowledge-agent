"""Streamlit views for persisted claim evidence and retrieval indexes."""

from __future__ import annotations

import html

import streamlit as st

from knowledge_agent.claims.config import ClaimSettings
from knowledge_agent.claims.dependencies import rebuild_claim_knowledge_base
from knowledge_agent.claims.lightrag import (
    load_lightrag_graph,
    validate_lightrag_index,
)
from knowledge_agent.claims.models import (
    DocumentMetadata,
    KnowledgeBaseEngine,
)
from knowledge_agent.claims.store import ClaimStore, get_document, get_page
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import load_llm_settings
from knowledge_agent.research.history import load_research_history
from knowledge_agent.ui.claims import ClaimEntry


def render_knowledge_base(
    entry: ClaimEntry,
    claim_settings: ClaimSettings,
    store: ClaimStore,
) -> None:
    _render_claim_metrics(store)
    _render_rebuild(entry, claim_settings, store)
    if not store.documents:
        st.info("This claim has no documents.")
        return

    _render_claim_overview(store)
    if "lightrag" in store.manifest.available_retrieval_modes:
        _render_lightrag_graph(entry)


def _render_claim_metrics(store: ClaimStore) -> None:
    manifest = store.manifest
    columns = st.columns(4)
    columns[0].metric("Claim", manifest.claim_id)
    columns[1].metric("Documents", len(manifest.documents))
    columns[2].metric("Chunks", manifest.chunk_count)
    columns[3].metric("Retrieval", " + ".join(manifest.available_retrieval_modes))


def _render_rebuild(
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
            ["Custom", "LightRAG", "Both"],
            index=_knowledge_base_index(store),
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
        target: KnowledgeBaseEngine
        if target_label == "LightRAG":
            target = "lightrag"
        elif target_label == "Both":
            target = "both"
        else:
            target = "custom"
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
            f"Rebuilt {manifest.claim_id} with "
            f"{' + '.join(manifest.available_retrieval_modes)} retrieval."
        )
        st.rerun()


def _knowledge_base_index(store: ClaimStore) -> int:
    modes = store.manifest.available_retrieval_modes
    if len(modes) > 1:
        return 2
    return 1 if modes[0] == "lightrag" else 0


def _render_lightrag_graph(entry: ClaimEntry) -> None:
    st.subheader("LightRAG graph")
    try:
        index_path = entry.path / "index" / "lightrag"
        metadata = validate_lightrag_index(index_path, entry.manifest)
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
        parties = party_rows(store)
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
    st.dataframe(document_rows(store), width="stretch", hide_index=True)

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


def document_rows(store: ClaimStore) -> list[dict[str, str | int]]:
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
    chunks = [chunk for chunk in store.chunks if chunk.document_id == document_id]
    if filter_text.strip():
        needle = filter_text.casefold()
        chunks = [
            chunk
            for chunk in chunks
            if needle in chunk.text.casefold() or needle in chunk.source_ref.casefold()
        ]
    if not chunks:
        st.info("No chunks match this filter.")
        return

    st.dataframe(
        [
            {
                "Source ref": chunk.source_ref,
                "Pages": ", ".join(chunk.page_ids),
                "Text": chunk.text,
            }
            for chunk in chunks
        ],
        width="stretch",
        hide_index=True,
    )
    selected_ref = st.selectbox(
        "Inspect chunk",
        [chunk.source_ref for chunk in chunks],
        key=f"chunk_select_{store.manifest.claim_id}_{document_id}",
    )
    selected_chunk = next(
        chunk for chunk in chunks if chunk.source_ref == selected_ref
    )
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


def _render_timeline(store: ClaimStore) -> None:
    events = timeline_rows(store)
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


def timeline_rows(store: ClaimStore) -> list[dict[str, str]]:
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


def party_rows(store: ClaimStore) -> list[dict[str, str]]:
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
