"""Claim research tool adapters."""

from __future__ import annotations

from knowledge_agent.agents.claim_researcher.models import EvidenceItem, ResearchQuery
from knowledge_agent.claims.store import ClaimStore, search_claim


def claim_search(
    store: ClaimStore,
    query: ResearchQuery,
    top_k: int,
) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            document_id=item.document_id,
            document_type=item.document_type,
            document_title=item.document_title,
            page_ids=item.page_ids,
            source_ref=item.source_ref,
            text=item.text,
        )
        for item in search_claim(store, query.query, top_k=top_k)
    ]
