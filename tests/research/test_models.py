import pytest
from pydantic import ValidationError

from knowledge_agent.agents.claim_researcher.models import ClaimResearchOutput
from knowledge_agent.agents.claim_researcher.validation import (
    validate_research_output,
)


FIRST = "CLM-001/DOC-001#DOC-001-CHUNK-001"
SECOND = "CLM-001/DOC-002#DOC-002-CHUNK-001"


def test_output_requires_nonempty_answer_and_no_sources_when_insufficient():
    with pytest.raises(ValidationError):
        ClaimResearchOutput(answer=" ", source_refs=[], evidence_sufficient=False)
    with pytest.raises(ValidationError, match="cannot declare sources"):
        ClaimResearchOutput(
            answer=f"Maybe. [{FIRST}]",
            source_refs=[FIRST],
            evidence_sufficient=False,
        )


def test_citations_must_match_inline_first_appearance_order():
    output = ClaimResearchOutput(
        answer=f"First. [{FIRST}] Second. [{SECOND}] Again. [{FIRST}]",
        source_refs=[SECOND, FIRST],
        evidence_sufficient=True,
    )
    with pytest.raises(ValueError, match="first-appearance order"):
        validate_research_output(output, {FIRST, SECOND})


def test_citations_must_come_from_current_run_retrieval():
    output = ClaimResearchOutput(
        answer=f"First. [{FIRST}]",
        source_refs=[FIRST],
        evidence_sufficient=True,
    )
    with pytest.raises(ValueError, match="outside retrieved evidence"):
        validate_research_output(output, set())


def test_evidence_sufficient_answer_requires_a_citation():
    output = ClaimResearchOutput(
        answer="A factual answer.",
        source_refs=[],
        evidence_sufficient=True,
    )
    with pytest.raises(ValueError, match="requires at least one citation"):
        validate_research_output(output, set())


def test_valid_output_is_returned_unchanged():
    output = ClaimResearchOutput(
        answer=f"First. [{FIRST}] Second. [{SECOND}]",
        source_refs=[FIRST, SECOND],
        evidence_sufficient=True,
    )
    assert validate_research_output(output, {FIRST, SECOND}) is output
