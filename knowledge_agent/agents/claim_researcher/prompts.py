"""Instructions for the claim-scoped deep research agent."""

from __future__ import annotations

from knowledge_agent.agents.claim_researcher.models import ClaimResearchPlan
from knowledge_agent.claims.models import DocumentMetadata


CLAIM_RESEARCH_INSTRUCTIONS = """
You research exactly one insurance claim knowledge base.

Plan material questions with the todo tools, then use claim_search to retrieve
claim evidence. Search iteratively when the first results leave a material gap.
Use only claim_search results as factual evidence. You have no web, filesystem,
shell, memory, or delegation access.

Every factual statement in an evidence-sufficient answer must cite an exact
source_ref in square brackets. Copy source_ref values verbatim and never append
page IDs. source_refs must list each inline citation once, in first-appearance
order. If the claim lacks enough evidence, say so, set evidence_sufficient to
false, and return no source_refs. Finish todos before producing the answer.
""".strip()


def _document_catalog(documents: list[DocumentMetadata]) -> str:
    return "\n".join(
        f"- {document.id} | {document.document_type} | "
        f"{document.title} | {document.summary}"
        for document in documents
    ) or "(no documents)"


def build_planning_instructions(
    documents: list[DocumentMetadata],
    *,
    clarification_round: int,
    max_clarifications: int,
) -> str:
    final_round = clarification_round >= max_clarifications
    decision = (
        "You have reached the clarification limit. State any remaining assumptions "
        "and return a ClaimResearchPlan now."
        if final_round
        else (
            "Return ResearchClarification only when one material ambiguity would "
            "change the research scope. Otherwise return ClaimResearchPlan."
        )
    )
    return (
        "You are the planning phase of a claim-scoped research agent. Verify the "
        "request, establish its scope, and design searches against only the listed "
        "claim documents. Do not answer the claim question and do not invent facts. "
        "You have todo tools for organizing the planning work, but no retrieval, "
        "web, filesystem, memory, or delegation tools. Ask at most one concise "
        "clarification question per response. "
        f"{decision}\n\nAvailable claim documents:\n{_document_catalog(documents)}"
    )


def build_research_instructions(
    documents: list[DocumentMetadata],
    approved_plan: ClaimResearchPlan | None = None,
) -> str:
    instructions = (
        f"{CLAIM_RESEARCH_INSTRUCTIONS}\n\n"
        "Available claim documents:\n"
        f"{_document_catalog(documents)}"
    )
    if approved_plan is not None:
        instructions += (
            "\n\nThe user approved this research plan. Execute it, adapting searches "
            "only when retrieved evidence reveals a material gap:\n"
            f"{approved_plan.model_dump_json(indent=2)}"
        )
    return instructions
