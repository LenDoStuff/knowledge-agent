import pytest
from pydantic import ValidationError

from claim_kb.schemas import DocumentEvent, DocumentParty


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
    )
    year_month = DocumentEvent(
        year=2026,
        month=6,
        sentence="Repairs were scheduled for June 2026.",
    )
    year_only = DocumentEvent(year=2026, sentence="The policy year was 2026.")
    no_date = DocumentEvent(sentence="The loss was reported.")

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
        DocumentEvent(month=13, sentence="The loss was reported.")

    with pytest.raises(ValidationError):
        DocumentEvent(day=32, sentence="The loss was reported.")

    with pytest.raises(ValidationError):
        DocumentEvent(sentence="")
