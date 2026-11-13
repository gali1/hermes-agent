"""C6 / C8 / D12: topic discovery, conflict resolution state, diagnostics.

Each test fails on the pre-change store: ``topics()`` returned a
``memory_type`` histogram, contradiction links had no dismissible state, and
``diagnostics()`` did not exist.
"""

from agent.memory.store import MemoryStore


def make_store(tmp_path):
    return MemoryStore(str(tmp_path / "memories.db"))


# ── C6: real topic discovery ──────────────────────────────────────────────

def test_topics_discover_content_terms_not_memory_types(tmp_path):
    store = make_store(tmp_path)
    for text in (
        "Kubernetes deployment rollout blocked on an image pull",
        "Kubernetes deployment rollout retried with a cached image",
        "Billing invoices reconciled through the nightly ledger service",
    ):
        store.store(text, memory_type="fact", project="p")

    topics = store.topics("p", limit=10)
    assert topics, "topics() must discover terms"
    names = [entry["topic"] for entry in topics]
    assert "kubernetes" in names
    assert "deployment" in names
    # The old implementation returned the type histogram; that is not a topic.
    assert "fact" not in names
    assert all(entry["kind"] == "term" for entry in topics)
    assert all(entry["latest"] for entry in topics)

    by_name = {entry["topic"]: entry for entry in topics}
    assert by_name["kubernetes"]["count"] == 2
    # Deterministic ordering: same store, same answer.
    assert store.topics("p", limit=10) == topics

    # Sorted by document frequency desc, then term asc.
    assert topics == sorted(topics, key=lambda entry: (-entry["count"], entry["topic"]))


def test_topics_are_bounded_and_project_scoped(tmp_path):
    store = make_store(tmp_path)
    for i in range(6):
        store.store(f"Alpha subsystem telemetry note {i}", project="p")
    store.store("Zulu billing reconciliation note", project="other")

    scoped = store.topics("p", limit=3)
    assert len(scoped) == 3
    assert all(entry["topic"] != "zulu" for entry in scoped)

    # Scan window bounds the work: with a two-row window, only those two count.
    shallow = store.topics("p", limit=10, scan_limit=2)
    assert all(entry["count"] <= 2 for entry in shallow)


# ── C8: contradiction links are dismissible, not deletable ────────────────

def test_conflict_can_be_dismissed_without_deleting_history(tmp_path):
    store = make_store(tmp_path)
    original = store.store(
        "The API gateway uses mutual TLS for all traffic", project="p"
    )["memory_id"]
    conflicting = store.store(
        "The API gateway does not use mutual TLS for all traffic", project="p"
    )["memory_id"]

    open_conflicts = store.get_conflicts("p")
    assert len(open_conflicts) == 1
    assert open_conflicts[0]["state"] == "open"
    assert open_conflicts[0]["confidence"] > 0

    resolved = store.resolve_conflict(
        conflicting, original, note="both true in different clusters"
    )
    assert resolved["success"] is True
    assert resolved["state"] == "dismissed"

    # Gone from the default view...
    assert store.get_conflicts("p") == []
    # ...but still there, with its history, when asked for.
    history = store.get_conflicts("p", include_resolved=True)
    assert len(history) == 1
    assert history[0]["state"] == "dismissed"
    assert history[0]["resolved_at"]
    assert history[0]["resolution_note"] == "both true in different clusters"

    # Nothing was deleted: link row and both memories survive.
    assert store.db.execute(
        "SELECT COUNT(*) FROM memory_links WHERE relation = 'contradicts'"
    ).fetchone()[0] == 1
    assert store.get(original) is not None
    assert store.get(conflicting) is not None

    # Reopening is possible; the state is a flag, not a deletion.
    reopened = store.resolve_conflict(conflicting, original, resolution="open")
    assert reopened["state"] == "open"
    assert len(store.get_conflicts("p")) == 1


def test_resolve_conflict_validates_input(tmp_path):
    store = make_store(tmp_path)
    first = store.store("Alpha gateway enables compression", project="p")["memory_id"]
    second = store.store("Beta gateway disables compression", project="p")["memory_id"]

    assert store.resolve_conflict(first, second)["success"] is False   # no link
    store.link(first, second, "related_to")
    assert store.resolve_conflict(first, second)["success"] is False   # not a conflict
    store.link(first, second, "contradicts")
    assert store.resolve_conflict(first, second, resolution="nonsense")["success"] is False


# ── D12: read-only diagnostics ────────────────────────────────────────────

def test_diagnostics_reports_health_without_repairing(tmp_path):
    store = make_store(tmp_path)
    first = store.store("Alpha gateway routing table snapshot", project="p")["memory_id"]
    second = store.store("Beta queue backlog alerting policy", project="p")["memory_id"]
    store.link(first, second, "related_to")

    healthy = store.diagnostics()
    assert healthy["read_only"] is True
    assert healthy["ok"] is True
    assert healthy["integrity"]["ok"] is True
    assert healthy["orphan_links"] == {"count": 0, "sample": []}
    assert healthy["stale_fts"]["ok"] is True
    assert healthy["stale_fts"]["memory_rows"] == 2
    assert healthy["per_project_growth"]["p"] == {"total": 2, "last_7d": 2, "last_24h": 2}
    assert healthy["schema"] == {"occurred_start": True, "occurred_end": True,
                                 "confidence": True}

    # Inject an orphan link (foreign keys off, as a crash would leave it) and a
    # stale FTS row (index entry removed, memory row still present).
    store.db.execute("PRAGMA foreign_keys=OFF")
    store.db.execute(
        "INSERT INTO memory_links (from_id, to_id, relation, metadata, created_at) "
        "VALUES ('ghost-a', 'ghost-b', 'related_to', '{}', '2026-01-01 00:00:00')"
    )
    store.db.commit()
    store.db.execute("PRAGMA foreign_keys=ON")
    row = store.db.execute(
        "SELECT rowid, content, tags, project FROM memories WHERE id = ?", (first,)
    ).fetchone()
    store.db.execute(
        "INSERT INTO memories_fts(memories_fts, rowid, content, tags, project) "
        "VALUES ('delete', ?, ?, ?, ?)",
        tuple(row),
    )
    store.db.commit()

    broken = store.diagnostics()
    assert broken["ok"] is False
    assert broken["orphan_links"]["count"] == 1
    assert broken["orphan_links"]["sample"][0]["from_id"] == "ghost-a"
    assert broken["stale_fts"]["ok"] is False
    assert broken["stale_fts"]["unindexed_rows"] == 1

    # Read-only: a second call sees the same problems (no repair side effect).
    again = store.diagnostics()
    assert again["orphan_links"]["count"] == 1
    assert again["stale_fts"]["unindexed_rows"] == 1
    assert again["stale_fts"]["fts_rows"] == broken["stale_fts"]["fts_rows"]
    assert store.health()["total_memories"] == 2


def test_diagnostics_on_empty_store(tmp_path):
    diag = make_store(tmp_path).diagnostics()
    assert diag["ok"] is True
    assert diag["stale_fts"]["ok"] is True
    assert diag["per_project_growth"] == {}
