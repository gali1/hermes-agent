"""Tests for the additive enhanced memory backend facade."""

import pytest

import agent.memory.backend as backend_module
from agent.memory.backend import SYSTEM_PROMPT_BLOCK, EnhancedMemoryBackend


def test_disabled_backend_is_noop(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": False})
    backend.initialize("s1", platform="cli")

    assert backend.available is False
    assert backend.prefetch("anything") == ""
    assert backend.system_prompt_block() == ""
    assert backend.health()["available"] is False
    assert backend.conflicts() == []
    assert backend.search("anything")["results"] == []
    assert backend.remember("some fact")["success"] is False
    backend.observe_turn("I prefer tabs", "ok")
    backend.on_session_end([])
    backend.on_pre_compress([])
    backend.shutdown()


def test_enabled_backend_roundtrip(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True, "max_recall": 5})
    backend.initialize("s1", platform="cli")
    assert backend.available is True

    backend.observe_turn(
        "I prefer concise answers and always use absolute paths",
        "Noted.",
        session_id="s1",
    )
    ctx = backend.prefetch("concise answers")
    assert "Recalled memory" in ctx
    assert "concise" in ctx

    health = backend.health()
    assert health["available"] is True
    assert health["total_memories"] >= 1
    assert health["db_path"].startswith(str(tmp_path))
    backend.shutdown()


def test_protocol_block(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    assert backend.system_prompt_block() == SYSTEM_PROMPT_BLOCK
    assert "<memory_system>" in SYSTEM_PROMPT_BLOCK
    backend.shutdown()


def test_secret_is_not_mirrored(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    backend.on_memory_write("add", "memory", "API key sk-abcdefghijklmnop123456")
    assert backend.health()["total_memories"] == 0
    backend.shutdown()


def test_memory_write_mirrors_curated(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    backend.on_memory_write("add", "user", "The user works in the hermes-agent repository")
    assert backend.health()["total_memories"] == 1
    backend.shutdown()


def test_auto_recall_can_be_disabled(tmp_path):
    backend = EnhancedMemoryBackend(
        str(tmp_path), {"enabled": True, "auto_recall": False}
    )
    backend.initialize("s1")
    backend.observe_turn("The deployment uses blue-green releases", "ok")
    assert backend.prefetch("deployment") == ""
    backend.shutdown()


def test_graceful_degradation_when_store_unavailable(tmp_path, monkeypatch):
    def _boom(path):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(backend_module, "MemoryStore", _boom)
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")

    assert backend.available is False
    assert backend.prefetch("anything") == ""
    assert backend.health()["available"] is False
    # Every hook must stay a safe no-op after degradation.
    backend.observe_turn("a fact", "ok")
    backend.on_session_end([])
    backend.on_memory_write("add", "memory", "a fact")
    backend.shutdown()


def test_pre_compress_extracts_durable_turns(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    messages = [
        {"role": "user", "content": "We decided to use PostgreSQL 17 for the analytics database"},
        {"role": "assistant", "content": "Understood."},
    ]
    assert backend.on_pre_compress(messages) == ""
    assert backend.health()["total_memories"] >= 1
    backend.shutdown()


def test_observe_turn_ignores_full_history(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    older = "An older durable fact about beta that must not be mined again"
    backend.observe_turn(
        "The current turn mentions alpha",
        "ok",
        messages=[
            {"role": "user", "content": older},
            {"role": "user", "content": "The current turn mentions alpha"},
        ],
    )
    assert backend.store.has_content("The current turn mentions alpha") is True
    assert backend.store.has_content(older) is False
    backend.shutdown()


def test_session_end_does_not_inflate_evidence(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    messages = [
        {"role": "user", "content": "We decided to use PostgreSQL 17 for the database"},
    ]
    backend.observe_turn(messages[0]["content"], "ok", session_id="s1")
    memory_id = backend.search("PostgreSQL")["results"][0]["id"]
    before = backend.store.get(memory_id)["proof_count"]

    backend.on_session_end(messages)
    after = backend.store.get(memory_id)["proof_count"]
    assert after == before
    backend.shutdown()
