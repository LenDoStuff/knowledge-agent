import pytest
from pydantic import ValidationError

from claim_kb.schemas import (
    DocumentChunk,
    DocumentEvent,
    DocumentParty,
    PageRange,
    PageText,
    StructuredClaimFile,
)


def test_document_party_requires_name_and_role():
    assert DocumentParty(name="Casey Sample", role="insured").role == "insured"

    with pytest.raises(ValidationError):
        DocumentParty(name="", role="insured")

    with pytest.raises(ValidationError):
        DocumentParty(name="Casey Sample", role="")


def test_document_event_accepts_full_and_partial_numeric_dates():
    full_date = DocumentEvent(
        year=2026,
        month=6,
        day=1,
        sentence="The loss happened on June 1, 2026.",
        source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
    )
    year_month = DocumentEvent(
        year=2026,
        month=6,
        sentence="Repairs were scheduled for June 2026.",
        source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
    )
    year_only = DocumentEvent(
        year=2026,
        sentence="The policy year was 2026.",
        source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
    )
    no_date = DocumentEvent(
        sentence="The loss was reported.",
        source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
    )

    assert full_date.year == 2026
    assert full_date.month == 6
    assert full_date.day == 1
    assert year_month.day is None
    assert year_only.month is None
    assert no_date.year is None
    assert no_date.month is None
    assert no_date.day is None
    assert no_date.sentence == "The loss was reported."

    with pytest.raises(ValidationError):
        DocumentEvent(
            month=13,
            sentence="The loss was reported.",
            source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
        )

    with pytest.raises(ValidationError):
        DocumentEvent(
            day=32,
            sentence="The loss was reported.",
            source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
        )

    with pytest.raises(ValidationError):
        DocumentEvent(
            sentence="",
            source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
        )

    with pytest.raises(ValidationError):
        DocumentEvent(sentence="The loss was reported.", source_ref="")


def test_page_and_chunk_citation_fields_are_exact():
    page = PageText(
        claim_id="CLM-001",
        page_number=1,
        page_id="CLM-001:p1",
        text="First notice of loss",
    )
    chunk = DocumentChunk(
        claim_id="CLM-001",
        document_id="DOC-001",
        chunk_id="DOC-001-CHUNK-001",
        source_ref="CLM-001/DOC-001#DOC-001-CHUNK-001",
        chunk_index=0,
        document_type="fnol",
        page_range=PageRange(start_page=1, end_page=2),
        page_ids=["CLM-001:p1", "CLM-001:p2"],
        text="First notice of loss",
    )

    assert page.page_id == "CLM-001:p1"
    assert chunk.page_ids == ["CLM-001:p1", "CLM-001:p2"]

    with pytest.raises(ValidationError):
        PageText(
            claim_id="CLM-001",
            page_number=1,
            page_id="wrong",
        )

    with pytest.raises(ValidationError):
        DocumentChunk.model_validate(
            {**chunk.model_dump(), "source_ref": "wrong"}
        )


def test_manifest_without_embedding_mode_defaults_to_snowflake():
    manifest = StructuredClaimFile.model_validate(
        {
            "claim_id": "CLM-001",
            "root_path": "data/claims/CLM-001",
            "source_files": ["claim.pdf"],
            "documents": [],
            "chunk_count": 0,
            "vector_store_path": "index/chroma",
            "embedding_provider": "snowflake",
            "embedding_model": "legacy-model",
        }
    )

    assert manifest.embedding_mode == "snowflake"
