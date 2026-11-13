"""D1–D4: event time, per-record confidence, hard temporal filter.

Every test here fails on the pre-change store: the new columns are missing, the
``temporal_filter`` keyword does not exist, and temporal proximity is computed
from ``created_at`` only.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from agent.memory.store import MemoryStore

_NEW_COLUMNS = {"occurred_start", "occurred_end", "confidence"}


def make_store(tmp_path, **kwargs):
    return MemoryStore(str(tmp_path / "memories.db"), **kwargs)


def _columns(store):
    return {row[1] for row in store.db.execute("PRAGMA table_info(memories)")}


def _age(store, memory_id, days):
    """Backdate store time, so event time and store time disagree."""
    ts = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    store.db.execute(
        "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
        (ts, ts, memory_id),
    )
    store.db.commit()
    return memory_id


def _ts(days_ago=0, hours_ago=0):
    return (
        datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(days=days_ago, hours=hours_ago)
    ).strftime("%Y-%m-%d %H:%M:%S")


# ── D1: event time beside store time ──────────────────────────────────────

def test_event_time_columns_exist_and_roundtrip(tmp_path):
    store = make_store(tmp_path)
    assert _NEW_COLUMNS <= _columns(store)

    stored = store.store(
        "The outage review meeting happened on the Tuesday before the release",
        project="p",
        occurred_start="2026-03-03 14:00:00",
        occurred_end="2026-03-03 15:30:00",
    )
    memory = store.get(stored["memory_id"])
    assert memory["occurred_start"] == "2026-03-03 14:00:00"
    assert memory["occurred_end"] == "2026-03-03 15:30:00"
    # Store time is untouched and still present beside the event time.
    assert memory["created_at"] and memory["created_at"] != memory["occurred_start"]


def test_event_time_is_nullable_and_normalised(tmp_path):
    store = make_store(tmp_path)
    plain = store.get(store.store("A memory with no event date")["memory_id"])
    assert plain["occurred_start"] is None
    assert plain["occurred_end"] is None

    aware = datetime(2026, 5, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    stored = store.store("A memory dated with an aware datetime", occurred_start=aware)
    assert store.get(stored["memory_id"])["occurred_start"] == "2026-05-01 10:00:00"

    # update() can set event time after the fact.
    assert store.update(plain["id"], occurred_start="2026-05-02 09:00:00")["success"]
    assert store.get(plain["id"])["occurred_start"] == "2026-05-02 09:00:00"


# ── D2: per-record confidence ─────────────────────────────────────────────

def test_confidence_defaults_neutral_and_clamps(tmp_path):
    store = make_store(tmp_path)
    default = store.store("Neutral confidence memory about queues")
    assert default["confidence"] == 0.5
    assert store.get(default["memory_id"])["confidence"] == 0.5

    explicit = store.store("Confident memory about queues", confidence=0.8)
    assert store.get(explicit["memory_id"])["confidence"] == 0.8

    high = store.store("Over-confident memory about queues", confidence=4.0)
    assert store.get(high["memory_id"])["confidence"] == 0.95

    low = store.store("Under-confident memory about queues", confidence=-1)
    assert store.get(low["memory_id"])["confidence"] == 0.0

    assert store.update(explicit["memory_id"], confidence=0.65)["success"]
    assert store.get(explicit["memory_id"])["confidence"] == 0.65


# ── D3: hard temporal filter ──────────────────────────────────────────────

def test_temporal_filter_is_a_hard_predicate(tmp_path):
    store = make_store(tmp_path)
    stale = store.store("Gateway deployment incident review notes", project="p")["memory_id"]
    fresh = store.store("Gateway deployment latency spike investigation", project="p")["memory_id"]
    _age(store, stale, 400)

    query = "gateway deployment today"

    # Score-only (the pre-existing behaviour) still returns the old row: the
    # window may only change its score.
    scored = store.search(query, limit=10, project="p", fusion="rrf", temporal=True)
    assert scored["retrieval"]["temporal_window"] is not None
    assert stale in {row["id"] for row in scored["results"]}

    # Hard filter: the window becomes a predicate over the candidate set.
    filtered = store.search(
        query, limit=10, project="p", fusion="rrf", temporal=True, temporal_filter=True
    )
    ids = {row["id"] for row in filtered["results"]}
    assert fresh in ids
    assert stale not in ids
    assert all(row["id"] != stale for row in filtered["results"])


def test_temporal_filter_respects_other_filters(tmp_path):
    store = make_store(tmp_path)
    keep = store.store("Ledger reconciliation ran early today", project="p")["memory_id"]
    store.store("Ledger reconciliation ran early today too", project="other")

    hits = store.search(
        "ledger reconciliation today", limit=10, project="p",
        fusion="rrf", temporal=True, temporal_filter=True,
    )
    assert [row["id"] for row in hits["results"]] == [keep]


# ── D4: event time vs store time ──────────────────────────────────────────

def test_temporal_proximity_uses_event_time_when_present(tmp_path):
    store = make_store(tmp_path)
    # Written 60 days ago, but the event happened 2 days ago.
    event = store.store(
        "Deployment window shifted to Thursday night",
        project="p",
        occurred_start=_ts(days_ago=2),
    )["memory_id"]
    _age(store, event, 60)

    # Written 60 days ago and never backfilled with an event time.
    store_time = store.store(
        "Deployment checklist reviewed for the release", project="p"
    )["memory_id"]
    _age(store, store_time, 60)

    result = store.search(
        "deployment last 7 days", limit=10, project="p", fusion="rrf", temporal=True,
        track_access=False,
    )
    assert result["retrieval"]["temporal_window"] is not None
    by_id = {row["id"]: row for row in result["results"]}
    assert event in by_id and store_time in by_id
    # Event time wins: the fresh event scores, the store-time row does not.
    assert by_id[event]["temporal_score"] > 0.5
    assert by_id[store_time]["temporal_score"] == 0.0

    # ...and the hard filter agrees, keeping the event even though the row was
    # written 60 days ago.
    filtered = store.search(
        "deployment last 7 days", limit=10, project="p", fusion="rrf",
        temporal=True, temporal_filter=True, track_access=False,
    )
    ids = {row["id"] for row in filtered["results"]}
    assert event in ids
    assert store_time not in ids


# ── Migrations: fresh DB and pre-change DB ────────────────────────────────

_OLD_SCHEMA = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'fact', project TEXT,
    scope TEXT NOT NULL DEFAULT 'global', session_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '', wing TEXT, room TEXT,
    tags TEXT NOT NULL DEFAULT '[]', importance REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active', valid_from TEXT, valid_until TEXT,
    source TEXT, metadata TEXT NOT NULL DEFAULT '{}', superseded_by TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0, last_accessed_at TEXT);
CREATE TABLE memory_links (
    from_id TEXT NOT NULL, to_id TEXT NOT NULL, relation TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, relation));
CREATE TABLE memory_config (
    project TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (project, key));
INSERT INTO memories (id, content, content_hash, memory_type, created_at, updated_at)
VALUES ('legacy1', 'A memory written before this upgrade existed', 'hash-legacy1',
        'fact', '2019-06-01 00:00:00', '2019-06-01 00:00:00');
INSERT INTO memories (id, content, content_hash, memory_type, created_at, updated_at)
VALUES ('legacy2', 'Another pre-upgrade memory about parsers', 'hash-legacy2',
        'fact', '2020-01-02 00:00:00', '2020-01-02 00:00:00');
"""


def test_migrations_on_fresh_database(tmp_path):
    path = str(tmp_path / "fresh.db")
    store = MemoryStore(path)
    columns = _columns(store)
    assert _NEW_COLUMNS <= columns
    # Idempotent: opening the same file again adds nothing and changes nothing.
    store.close()
    reopened = MemoryStore(path)
    assert _columns(reopened) == columns
    assert reopened.health()["schema_version"] == 2
    reopened.store("A new memory with an event date", occurred_start="2026-02-02 10:00:00")
    assert reopened.search("event date", limit=5)["results"]


def test_migrations_on_existing_head_shape_database(tmp_path):
    """The exact pre-change shape: B1/B12 evidence columns, no new columns."""
    path = str(tmp_path / "head.db")
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'fact', project TEXT,
            scope TEXT NOT NULL DEFAULT 'global', session_id TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL DEFAULT '', wing TEXT, room TEXT,
            tags TEXT NOT NULL DEFAULT '[]', importance REAL NOT NULL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'active', valid_from TEXT, valid_until TEXT,
            source TEXT, metadata TEXT NOT NULL DEFAULT '{}',
            proof_count INTEGER NOT NULL DEFAULT 1, last_reinforced_at TEXT,
            superseded_by TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0, last_accessed_at TEXT);
        CREATE TABLE memory_links (
            from_id TEXT NOT NULL, to_id TEXT NOT NULL, relation TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
            PRIMARY KEY (from_id, to_id, relation));
        CREATE TABLE memory_config (
            project TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY (project, key));
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            content, tags, project, content='memories', content_rowid='rowid');
        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags, project)
            VALUES (new.rowid, new.content, new.tags, new.project);
        END;
        INSERT INTO memories
            (id, content, content_hash, memory_type, project, proof_count,
             last_reinforced_at, created_at, updated_at)
        VALUES ('head1', 'A memory written two commits before this change',
                'hash-head1', 'fact', 'p', 3, '2026-10-01 00:00:00',
                '2026-10-01 00:00:00', '2026-10-01 00:00:00');
        """
    )
    old.commit()
    old.close()

    store = MemoryStore(path)
    columns = _columns(store)
    assert _NEW_COLUMNS <= columns
    row = store.get("head1")
    assert row["proof_count"] == 3          # existing evidence preserved
    assert row["confidence"] == 0.5         # new column takes its default
    assert row["occurred_start"] is None
    assert row["occurred_end"] is None
    # FTS still answers for the migrated row, and the hard filter falls back to
    # store time because no event time exists.
    assert store.search("two commits", limit=5)["results"]
    assert {r["id"] for r in store.search(
        "commits in 2026", limit=5, fusion="rrf", temporal=True, temporal_filter=True
    )["results"]} == {"head1"}
    # New writes work on the migrated file.
    assert store.store("A brand new memory after the migration",
                       occurred_start="2026-11-01 00:00:00")["success"]
    store.close()
    assert MemoryStore(path).get("head1")["proof_count"] == 3


def test_migrations_on_pre_change_database(tmp_path):
    path = str(tmp_path / "legacy.db")
    old = sqlite3.connect(path)
    old.executescript(_OLD_SCHEMA)
    old.commit()
    old.close()

    store = MemoryStore(path)
    columns = _columns(store)
    assert _NEW_COLUMNS <= columns
    # Idempotent: re-running the migration list is a no-op and adds nothing.
    store._run_migrations()
    store._run_migrations()
    assert _columns(store) == columns

    legacy = store.get("legacy1")
    assert legacy is not None
    assert legacy["content"] == "A memory written before this upgrade existed"
    assert legacy["confidence"] == 0.5          # neutral default, not NULL
    assert legacy["occurred_start"] is None     # nullable, falls back to store time
    assert legacy["occurred_end"] is None
    assert legacy["proof_count"] == 1

    # The old rows stay retrievable by every path, including the temporal one.
    assert store.search("upgrade", limit=5)["results"]
    assert store.search("parsers", limit=5, fusion="rrf", temporal=True)["results"]
    hard = store.search(
        "parsers in 2020", limit=5, fusion="rrf", temporal=True, temporal_filter=True
    )
    # 2020 window matches on created_at because no event time is stored.
    assert {row["id"] for row in hard["results"]} == {"legacy2"}

    # New writes and a second open both work on the migrated file.
    store.store("Post-upgrade memory about parsers", project="p",
                occurred_start="2026-01-01 00:00:00")
    store.close()
    reopened = MemoryStore(path)
    assert _columns(reopened) == columns
    assert reopened.get("legacy1") is not None
    assert reopened.search("parsers", limit=5, project="p")["results"]
