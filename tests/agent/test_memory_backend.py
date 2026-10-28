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


def test_maybe_backup_snapshots_the_database(tmp_path):
    """Archiving never leaves the file, so the file itself needs a copy."""
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True, "backup_keep": 3})
    backend.initialize("s1", platform="cli")
    backend.remember("The gateway owns the websocket transport layer")

    first = backend.maybe_backup(force=True)
    assert first["success"] is True
    assert first["bytes"] > 0
    assert first["path"].startswith(str(tmp_path / "backups" / "memory"))

    import sqlite3
    snapshot = sqlite3.connect(first["path"])
    try:
        assert snapshot.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        snapshot.close()

    # The interval marker makes a second unforced pass a no-op.
    second = backend.maybe_backup()
    assert second["skipped"] == "interval"
    backend.shutdown()


def test_maybe_backup_prunes_to_keep(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True, "backup_keep": 2})
    backend.initialize("s1", platform="cli")
    backend.remember("Retention bookkeeping lives in a sidecar marker file")

    backup_dir = tmp_path / "backups" / "memory"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        (backup_dir / f"memories-2020010{index}-000000.db").write_bytes(b"stale")

    result = backend.maybe_backup(force=True)
    assert result["success"] is True
    remaining = sorted(name for name in __import__("os").listdir(backup_dir)
                       if name.startswith("memories-"))
    assert len(remaining) == 2, remaining
    # The newest snapshot survives pruning.
    assert remaining[-1] == __import__("os").path.basename(result["path"])
    backend.shutdown()


def test_backfill_origin_tiers_rows_written_by_old_code(tmp_path):
    """A process running pre-tiering code leaves rows retention cannot classify."""
    import sqlite3

    state_db = tmp_path / "state.db"
    connection = sqlite3.connect(state_db)
    connection.execute("CREATE TABLE messages (role TEXT, content TEXT)")
    connection.executemany(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        [
            ("user", "The cache is disabled for the session store"),
            ("assistant", "Fixed and verified, the cache setting now sticks"),
        ],
    )
    connection.commit()
    connection.close()

    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1", platform="cli")
    store = backend.store
    untagged_user = store.store("The cache is disabled for the session store",
                                importance=0.5, source={"type": "conversation"})["memory_id"]
    untagged_reply = store.store("Fixed and verified, the cache setting now sticks",
                                importance=0.5, source={"type": "conversation"})["memory_id"]
    curated = store.store("A note written through the memory tool",
                          importance=0.7, tags=["origin:curated"])["memory_id"]

    counts = backend.backfill_origin()
    assert counts["tagged"] == 2, counts
    assert counts["user"] == 1 and counts["assistant"] == 1

    user_row = store.get(untagged_user, track_access=False)
    reply_row = store.get(untagged_reply, track_access=False)
    assert user_row["importance"] == 0.6
    assert "origin:user" in user_row["tags"]
    assert reply_row["importance"] == 0.4
    assert "origin:assistant" in reply_row["tags"]

    # Curated rows were already tagged and must not be rewritten.
    curated_row = store.get(curated, track_access=False)
    assert curated_row["tags"] == ["origin:curated"]
    backend.shutdown()


def test_session_end_purges_stored_junk(tmp_path):
    """The sweep is automatic: capture filters re-run over the file at session end."""
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    store = backend._store
    assert store is not None
    junk = store.store("All enhanced memory features working?", tags=["origin:user"])["memory_id"]
    keep = store.store(
        "Root cause: the ACP client sends its own cwd, which overrides the CLI argument.",
        tags=["origin:assistant"],
    )["memory_id"]

    backend.on_session_end([])

    assert store.get(junk, track_access=False)["status"] == "archived"
    assert store.get(keep, track_access=False)["status"] == "active"
    backend.shutdown()


def test_archived_junk_is_not_resurrected(tmp_path):
    """Session-end mining replays history; it must not un-archive a decision."""
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    store = backend._store
    assert store is not None
    text = "All enhanced memory features working?"
    mid = store.store(text, tags=["origin:user"])["memory_id"]

    assert backend.purge_junk()["archived"] == 1
    assert store.get(mid, track_access=False)["status"] == "archived"

    # The same turn arrives again at session end, and per-turn capture sees it too.
    backend.on_session_end([{"role": "user", "content": text}])
    backend.observe_turn(text, "yes", session_id="s1")

    assert store.get(mid, track_access=False)["status"] == "archived"
    active = store._execute(
        "SELECT COUNT(*) FROM memories WHERE status = 'active' AND content_hash = "
        "(SELECT content_hash FROM memories WHERE id = ?)", (mid,),
    ).fetchone()[0]
    assert active == 0
    backend.shutdown()


def test_junk_purge_can_be_disabled(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True, "junk_purge_enabled": False})
    backend.initialize("s1")
    store = backend._store
    assert store is not None
    junk = store.store("Is the gateway on port 8080?", tags=["origin:user"])["memory_id"]
    assert backend.purge_junk() == {"skipped": "disabled"}
    assert store.get(junk, track_access=False)["status"] == "active"
    backend.shutdown()
