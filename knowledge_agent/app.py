"""Compose the Streamlit claim-ingestion and research workbench."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from knowledge_agent.claims.config import ClaimSettings, load_claim_settings
from knowledge_agent.claims.models import KnowledgeBaseEngine
from knowledge_agent.claims.store import load_claim_store
from knowledge_agent.config import load_profile
from knowledge_agent.llm.config import load_llm_settings
from knowledge_agent.ui.claims import (
    COMBINED_LABEL,
    SEPARATE_LABEL,
    ClaimEntry,
    InvalidClaim,
    UploadMode,
    discover_claims,
    ingest_uploads,
    validate_uploads,
)
from knowledge_agent.ui.knowledge_base import render_knowledge_base
from knowledge_agent.ui.research import render_research


PAGE_STYLE = """
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
"""


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Claim Research Workbench", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
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
        store = load_claim_store(selected.path, validate_index=False)
    except Exception as exc:
        st.error(f"Could not open claim {selected.manifest.claim_id}: {exc}")
        return

    knowledge_tab, chat_tab = st.tabs(["Knowledge base", "Research chat"])
    with knowledge_tab:
        render_knowledge_base(selected, claim_settings, store)
    with chat_tab:
        render_research(selected, claim_settings, store)


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
        ["Custom", "LightRAG", "Both"],
        help=(
            "Custom uses the current lexical or Snowflake/Chroma index. "
            "LightRAG builds a claim-local entity graph and vector index. "
            "Both builds both indexes and lets each research run choose one."
        ),
    )
    knowledge_base: KnowledgeBaseEngine
    if knowledge_base_label == "LightRAG":
        knowledge_base = "lightrag"
    elif knowledge_base_label == "Both":
        knowledge_base = "both"
    else:
        knowledge_base = "custom"
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
    upload_list = (
        list(uploads or [])
        if mode == "separate"
        else ([uploads] if uploads else [])
    )
    if not st.button("Ingest claim", type="primary", width="stretch"):
        return

    try:
        llm_settings = load_llm_settings(load_profile())
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


if __name__ == "__main__":
    main()
