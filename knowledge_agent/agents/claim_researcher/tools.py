"""Claim-local retrieval adapter for the research agent."""

from __future__ import annotations

import asyncio

from knowledge_agent.agents.claim_researcher.models import EvidenceItem
from knowledge_agent.claims.store import ClaimStore, search_claim


async def search_claim_evidence(
    store: ClaimStore,
    query: str,
    top_k: int,
) -> list[EvidenceItem]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if store.manifest.retrieval_mode == "lightrag":
        if store.lightrag is None:
            raise RuntimeError("LightRAG claim store requires retrieval dependencies")
        chunk_ids = await store.lightrag.retrieve_chunk_ids(query, top_k)
        unknown = [chunk_id for chunk_id in chunk_ids if chunk_id not in store.chunks_by_id]
        if unknown:
            raise ValueError(f"LightRAG returned unknown claim chunks: {unknown}")
        results = [
            (
                store.chunks_by_id[chunk_id],
                store.documents_by_id[store.chunks_by_id[chunk_id].document_id],
            )
            for chunk_id in chunk_ids
        ]
        return [
            EvidenceItem(
                document_id=chunk.document_id,
                document_type=chunk.document_type,
                document_title=document.title,
                page_ids=chunk.page_ids,
                source_ref=chunk.source_ref,
                text=chunk.text,
            )
            for chunk, document in results
        ]

    custom_results = await asyncio.to_thread(
        search_claim,
        store,
        query,
        top_k=top_k,
    )
    return [
        EvidenceItem(
            document_id=item.document_id,
            document_type=item.document_type,
            document_title=item.document_title,
            page_ids=item.page_ids,
            source_ref=item.source_ref,
            text=item.text,
        )
        for item in custom_results
    ]
