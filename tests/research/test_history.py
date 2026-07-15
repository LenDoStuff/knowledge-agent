"""Tests for typed, atomic research-history persistence."""

from pathlib import Path

import pytest
from pydantic_ai import ModelRequest, UserPromptPart

from knowledge_agent.research.history import (
    InteractionUsage,
    KnowledgeBaseSnapshot,
    ResearchHistory,
    ResearchInteraction,
    clear_terminal_interactions,
    delete_interaction,
    load_research_history,
    research_history_path,
    store_interaction,
)


def interaction(claim_id: str = "CLM-001", *, status="planning"):
    return ResearchInteraction(
        claim_id=claim_id,
        status=status,
        question="What was repaired?",
        planning_enabled=True,
        agent_messages=[
            ModelRequest(parts=[UserPromptPart("What was repaired?")])
        ],
        error="failed" if status == "failed" else None,
    )


def test_missing_history_is_empty_and_round_trips_native_messages(tmp_path):
    history = load_research_history(tmp_path, "CLM-001")
    assert history == ResearchHistory(claim_id="CLM-001")

    item = interaction()
    stored = store_interaction(tmp_path, item)
    loaded = load_research_history(tmp_path, "CLM-001")

    assert stored.interactions[0].id == item.id
    assert loaded.interactions[0].agent_messages == item.agent_messages
    assert research_history_path(tmp_path).exists()
    assert not Path(f"{research_history_path(tmp_path)}.tmp").exists()


def test_history_round_trips_knowledge_base_snapshot(tmp_path):
    item = interaction()
    item.knowledge_base = KnowledgeBaseSnapshot(
        retrieval_mode="lightrag",
        embedding_provider="nvidia",
        embedding_model="baai/bge-m3",
        lightrag_version="1.5.4",
        lightrag_index_claim_id="CLM-001",
        lightrag_index_llm_provider="nvidia",
        lightrag_index_llm_model="provider/model",
        lightrag_embedding_dimension=1024,
        lightrag_embedding_max_tokens=8192,
        lightrag_query_mode="hybrid",
        lightrag_indexed_chunk_count=3,
        lightrag_entity_count=2,
        lightrag_relationship_count=1,
        lightrag_indexing_usage=InteractionUsage(requests=4, input_tokens=100),
    )

    store_interaction(tmp_path, item)
    loaded = load_research_history(tmp_path, "CLM-001").interactions[0]

    assert loaded.knowledge_base == item.knowledge_base


def test_existing_history_without_snapshot_remains_explicitly_unknown(tmp_path):
    item = interaction()
    payload = ResearchHistory(
        claim_id="CLM-001",
        interactions=[item],
    ).model_dump(mode="json")
    payload["interactions"][0].pop("knowledge_base")
    path = research_history_path(tmp_path)
    path.parent.mkdir(parents=True)
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_research_history(tmp_path, "CLM-001").interactions[0]
    assert loaded.knowledge_base is None


def test_history_rejects_claim_mismatch_and_multiple_active_records(tmp_path):
    path = research_history_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        ResearchHistory(claim_id="OTHER").model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        load_research_history(tmp_path, "CLM-001")

    first = interaction()
    path.write_text(ResearchHistory(claim_id="CLM-001").model_dump_json(), encoding="utf-8")
    store_interaction(tmp_path, first)
    with pytest.raises(ValueError, match="already has an active"):
        store_interaction(tmp_path, interaction())


def test_terminal_history_can_be_deleted_or_cleared(tmp_path):
    failed = interaction(status="failed")
    store_interaction(tmp_path, failed)

    history = delete_interaction(tmp_path, "CLM-001", failed.id)
    assert history.interactions == []

    active = interaction()
    store_interaction(tmp_path, active)
    with pytest.raises(ValueError, match="active"):
        delete_interaction(tmp_path, "CLM-001", active.id)
    active.status = "cancelled"
    store_interaction(tmp_path, active)

    store_interaction(tmp_path, interaction(status="cancelled"))
    history = clear_terminal_interactions(tmp_path, "CLM-001")
    assert history.interactions == []
