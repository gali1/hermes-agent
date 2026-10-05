"""Tests for the enhanced-memory actions on the built-in ``memory`` tool.

The enhanced actions are additive: existing add/replace/remove/batch behavior
is unchanged, and the enhanced actions return a configuration hint when the
additive layer is disabled.
"""

import json

from agent.memory.backend import EnhancedMemoryBackend
from tools.memory_tool import MEMORY_SCHEMA, MemoryStore as CuratedStore, memory_tool


def make_backend(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True, "max_recall": 5})
    backend.initialize("s1")
    return backend


def test_schema_declares_enhanced_actions_and_params():
    props = MEMORY_SCHEMA["parameters"]["properties"]
    for action in ("search", "recall", "conflicts", "timeline", "topics", "health", "reinforce"):
        assert action in props["action"]["enum"]
    for key in ("query", "limit", "memory_type", "graph_expand", "temporal", "memory_id"):
        assert key in props


def test_enhanced_actions_error_when_disabled():
    result = json.loads(memory_tool(action="search", query="x", store=None, backend=None))
    assert result.get("success") is False
    assert "Enhanced memory is not enabled" in result["error"]


def test_search_and_recall_roundtrip(tmp_path):
    backend = make_backend(tmp_path)
    backend.remember("The gateway uses mutual TLS for all traffic", memory_type="fact")

    search = json.loads(memory_tool(action="search", query="mutual TLS", backend=backend))
    assert search["count"] >= 1
    assert "mutual TLS" in search["results"][0]["content"]
    assert "retrieval" not in search

    recall = json.loads(memory_tool(action="recall", query="mutual TLS", backend=backend))
    assert recall["count"] >= 1
    assert "retrieval" in recall
    assert recall["retrieval"]["mode"] in ("rrf", "rrf-empty", "fallback")
    backend.shutdown()


def test_reinforce_increments_proof(tmp_path):
    backend = make_backend(tmp_path)
    memory_id = backend.remember("A durable fact about indexing")["memory_id"]
    before = backend.store.get(memory_id)["proof_count"]

    result = json.loads(memory_tool(action="reinforce", memory_id=memory_id, backend=backend))
    assert result["success"] is True
    assert result["proof_count"] == before + 1
    backend.shutdown()


def test_diagnostics_actions(tmp_path):
    backend = make_backend(tmp_path)
    backend.remember("Caching is enabled for the session store", project="p")
    backend.remember("Caching is disabled for the session store", project="p")

    conflicts = json.loads(memory_tool(action="conflicts", backend=backend))
    assert len(conflicts["conflicts"]) >= 1

    timeline = json.loads(memory_tool(action="timeline", backend=backend, limit=5))
    assert timeline["count"] >= 2

    topics = json.loads(memory_tool(action="topics", backend=backend))
    assert topics["topics"]

    health = json.loads(memory_tool(action="health", backend=backend))
    assert health["available"] is True
    assert health["total_memories"] >= 2
    backend.shutdown()


def test_missing_query_and_memory_id_errors(tmp_path):
    backend = make_backend(tmp_path)
    assert json.loads(memory_tool(action="search", backend=backend))["success"] is False
    assert json.loads(memory_tool(action="reinforce", backend=backend))["success"] is False
    backend.shutdown()


def test_curated_actions_still_route_to_store(tmp_path):
    backend = make_backend(tmp_path)
    store = CuratedStore()
    store.load_from_disk()

    result = json.loads(memory_tool(
        action="add", target="memory", content="A curated note", store=store, backend=backend
    ))
    assert result["success"] is True
    backend.shutdown()
