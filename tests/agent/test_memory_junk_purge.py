"""Algorithmic junk purge: the capture filters re-run over what is already stored.

Capture-side filtering only protects new turns; rows stored before the filters
existed still match every query through FTS prefix expansion.  These tests pin
the purge's two obligations: it archives the classes the filters reject, and it
never touches protected rows (curated writes, user scope, long documents).
"""

import json
import os

from agent.memory.store import MemoryStore


def make_store(tmp_path):
    return MemoryStore(str(tmp_path / "memories.db"))


def status_of(store, memory_id):
    row = store.get(memory_id, track_access=False)
    assert row is not None
    return row["status"]


def test_purge_archives_only_rejected_classes(tmp_path):
    store = make_store(tmp_path)

    junk = {
        "question": store.store("All enhanced memory features working?", tags=["origin:user"])["memory_id"],
        "command": store.store("list all skills available", tags=["origin:user"])["memory_id"],
        "fallback": store.store(
            "Your request was not processed. Send it again if you still want me to carry it out.",
            tags=["origin:assistant"],
        )["memory_id"],
        "status": store.store("Operation interrupted: waiting for model response (7.8s elapsed).",
                              tags=["origin:assistant"])["memory_id"],
    }
    keep = {
        "preference": store.store("I prefer using uv for Python environments and never pip install globally",
                                  tags=["origin:user"])["memory_id"],
        "root_cause": store.store(
            "Root cause: the ACP client sends its own cwd, which overrides the CLI argument at session start.",
            tags=["origin:assistant"],
        )["memory_id"],
        "curated": store.store("All enhanced memory features are described in the docs page",
                               tags=["origin:curated"])["memory_id"],
        "user_scope": store.store("Question: does the assistant answer in English?", scope="user")["memory_id"],
    }

    dry = store.purge_junk(dry_run=True)
    assert dry["dry_run"] is True and dry["archived"] == 0
    assert sorted(dry["candidates"]) == sorted(junk.values())
    for mid in junk.values():
        assert status_of(store, mid) == "active"

    done = store.purge_junk()
    assert done["success"] is True
    assert done["archived"] == len(junk)
    for mid in junk.values():
        assert status_of(store, mid) == "archived"
    for mid in keep.values():
        assert status_of(store, mid) == "active", mid


def test_purge_is_reversible_and_cold_exported(tmp_path):
    store = make_store(tmp_path)
    mid = store.store("All enhanced memory features working?", tags=["origin:user"])["memory_id"]

    done = store.purge_junk()
    assert done["exported"] >= 1
    assert os.path.exists(done["cold_export"])
    exported = [json.loads(line) for line in open(done["cold_export"], encoding="utf-8")]
    assert any(row["memory"]["id"] == mid and "features working" in row["memory"]["content"]
               for row in exported)

    # Content survives archiving, so one update restores it.
    row = store.get(mid, track_access=False)
    assert row is not None and "features working" in row["content"]
    meta = row["metadata"]
    assert isinstance(meta, dict) and meta["purge_reason"].startswith("purged")
    store.update(mid, status="active")
    assert status_of(store, mid) == "active"


def test_purge_ignores_long_documents(tmp_path):
    store = make_store(tmp_path)
    # A long note is a document that predates the capture cap, not a rejected
    # candidate — even when it ends in a question.
    long_note = (
        "Investigated the retention pass and confirmed it archives rather than deletes, "
        "writes a cold export first, and only then flips the row status. "
        + "Filler detail. " * 200
        + "Should the export raise, does it fail closed?"
    )
    assert len(long_note) > 3000
    mid = store.store(long_note, tags=["origin:assistant"])["memory_id"]
    assert store.purge_junk()["archived"] == 0
    assert status_of(store, mid) == "active"


def test_purge_limit_and_empty_store(tmp_path):
    store = make_store(tmp_path)
    assert store.purge_junk()["archived"] == 0
    for i in range(5):
        store.store(f"Is the gateway listening on port 808{i}?", tags=["origin:user"])
    assert len(store.purge_junk(dry_run=True, limit=2)["candidates"]) == 2
    assert store.purge_junk(limit=3)["archived"] == 3
    assert store.health()["active_memories"] == 2
