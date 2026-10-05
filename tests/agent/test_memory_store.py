"""Integration tests for the enhanced memory store (agent/memory/store.py).

Adapted from the OpenCode/MemPalace Rekal integration checks plus core-layer
extensions (scope/session filters, FTS self-heal, concurrency smoke).
"""

import sqlite3
import threading

from agent.memory.store import MemoryStore, _quote_fts


def make_store(tmp_path):
    return MemoryStore(str(tmp_path / "memories.db"))


# ── Baseline contract ─────────────────────────────────────────────────────

def test_baseline_unchanged(tmp_path):
    store = make_store(tmp_path)
    for text in ("Session storage uses SQLite WAL", "Retry uses exponential backoff"):
        store.store(text, project="p")

    result = store.search("storage", limit=5, project="p")
    assert sorted(result.keys()) == ["query", "results", "total_candidates", "weights"]
    assert "retrieval" not in result

    row = result["results"][0]
    for field in ("id", "content", "memory_type", "project", "scope", "session_id",
                  "tags", "importance", "created_at", "updated_at", "access_count",
                  "proof_count", "score", "fts_score", "vec_score", "recency_score",
                  "access_score", "source"):
        assert field in row

    assert "rrf_score" not in row
    assert set(result["weights"]) >= {"w_fts", "w_vec", "w_recency", "half_life"}

    assert _quote_fts("alpha beta") == '"alpha"* "beta"*'
    assert _quote_fts("alpha beta", match_any=True) == '"alpha"* OR "beta"*'


def test_lifecycle_operations(tmp_path):
    store = make_store(tmp_path)
    first = store.store("Original content about caching", project="p")["memory_id"]
    second = store.store("Second memory about indexing", project="p")["memory_id"]

    assert first
    assert store.get(first)["content"] == "Original content about caching"
    assert store.update(first, content="Updated content about caching")["success"]
    assert "Updated" in store.get(first)["content"]
    assert store.link(first, second, "related_to")["success"]
    assert len(store.related(first)) >= 1

    superseded = store.supersede(first, "Superseded content about caching", project="p")
    assert superseded.get("new_id")

    assert store.health()["total_memories"] >= 2
    assert isinstance(store.topics("p"), (dict, list))
    assert isinstance(store.timeline("p"), (dict, list))
    assert store.delete(second)["success"]
    assert store.get(second) is None

    # Superseded rows are excluded from search.
    hits = store.search("caching", limit=10, project="p")
    assert all(hit["id"] != first for hit in hits["results"])


# ── Migrations ────────────────────────────────────────────────────────────

def test_migration_adds_evidence_columns(tmp_path):
    path = str(tmp_path / "memories.db")
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
        VALUES ('legacy1', 'A memory written before the upgrade', 'hash-legacy', 'fact',
                '2020-01-01 00:00:00', '2020-01-01 00:00:00');
        """
    )
    old.commit()
    old.close()

    store = MemoryStore(path)
    columns = {row[1] for row in store.db.execute("PRAGMA table_info(memories)")}
    assert "proof_count" in columns
    assert "last_reinforced_at" in columns

    legacy = store.get("legacy1")
    assert legacy is not None and "before the upgrade" in legacy["content"]
    assert legacy.get("proof_count") == 1
    assert isinstance(store.search("upgrade", limit=5), dict)
    assert isinstance(store.search("upgrade", limit=5, fusion="rrf"), dict)

    reopened = MemoryStore(path)
    assert reopened.get("legacy1") is not None


# ── Evidence ──────────────────────────────────────────────────────────────

def test_proof_accumulation(tmp_path):
    store = make_store(tmp_path)
    text = "The deploy script requires an explicit region flag"

    first = store.store(text, project="p")
    assert not first.get("duplicate")

    second = store.store(text, project="p")
    assert second.get("duplicate") is True
    assert second["memory_id"] == first["memory_id"]
    assert second.get("proof_count") == 2

    third = store.store(text, project="p")
    assert third.get("proof_count") == 3

    stored = store.get(first["memory_id"])
    assert stored.get("proof_count") == 3
    assert stored.get("last_reinforced_at")
    assert store.db.execute(
        "SELECT COUNT(*) FROM memories WHERE content = ?", (text,)
    ).fetchone()[0] == 1


def test_reinforce_missing_memory(tmp_path):
    store = make_store(tmp_path)
    assert store.reinforce("nope")["success"] is False


# ── Contradiction ─────────────────────────────────────────────────────────

def test_contradiction_links(tmp_path):
    store = make_store(tmp_path)
    original = store.store("The API gateway uses mutual TLS for all traffic", project="p")
    assert not original.get("contradicts")

    conflicting = store.store("The API gateway does not use mutual TLS for all traffic", project="p")
    assert conflicting.get("contradicts")
    assert any(c["memory_id"] == original["memory_id"] for c in conflicting.get("contradicts", []))
    assert all(0.0 < c["confidence"] <= 0.95 for c in conflicting.get("contradicts", []))

    links = store.db.execute(
        "SELECT COUNT(*) FROM memory_links WHERE relation = 'contradicts'"
    ).fetchone()[0]
    assert links >= 1

    conflicts = store.get_conflicts("p")
    assert len(conflicts) >= 1
    assert conflicts[0]["confidence"] > 0

    assert store.get(original["memory_id"]) is not None
    assert store.get(conflicting["memory_id"]) is not None

    unrelated = store.store("Billing reconciliation runs nightly at 02:00 UTC", project="p")
    assert not unrelated.get("contradicts")


# ── Advanced retrieval ────────────────────────────────────────────────────

def test_rank_fusion(tmp_path):
    store = make_store(tmp_path)
    for text in ("Session storage uses SQLite with WAL mode",
                 "Retry policy uses exponential backoff",
                 "Auth tokens expire after thirty days"):
        store.store(text, project="p")

    fused = store.search("storage", limit=5, project="p", fusion="rrf")
    assert "retrieval" in fused
    assert fused["retrieval"]["mode"] in ("rrf", "rrf-empty")
    assert len(fused["results"]) >= 1

    if fused["retrieval"]["mode"] == "rrf":
        top = fused["results"][0]
        assert "rrf_score" in top
        assert top.get("rrf_rank") == 1
        assert isinstance(top.get("source_ranks"), dict)
        assert all(
            fused["results"][i]["score"] >= fused["results"][i + 1]["score"]
            for i in range(len(fused["results"]) - 1)
        )


def test_graph_expansion(tmp_path):
    store = make_store(tmp_path)
    orphan = store.store("Quorum timeouts were raised to nine seconds", project="p")["memory_id"]
    store.db.execute(
        "UPDATE memories SET created_at = '2020-01-01 00:00:00', "
        "updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        (orphan,),
    )
    for i in range(30):
        filler = store.store(
            f"Filler note number {i} about unrelated subsystem topics", project="p"
        )["memory_id"]
        ts = f"2026-01-{i + 1:02d} 00:00:00"
        store.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (ts, ts, filler),
        )
    hit = store.store("The zephyr subsystem handles quorum election", project="p")["memory_id"]
    store.db.execute(
        "UPDATE memories SET created_at = '2026-06-01 00:00:00', "
        "updated_at = '2026-06-01 00:00:00' WHERE id = ?",
        (hit,),
    )
    store.db.commit()
    store.link(hit, orphan, "related_to")

    without = store.search("zephyr", limit=3, project="p", fusion="rrf")
    assert orphan not in [r["id"] for r in without["results"]]

    with_graph = store.search("zephyr", limit=3, project="p", fusion="rrf", graph_expand=True)
    ids = [r["id"] for r in with_graph["results"]]
    assert with_graph["retrieval"].get("graph_expanded", 0) >= 1
    assert orphan in ids

    expanded = next((r for r in with_graph["results"] if r["id"] == orphan), None)
    assert expanded is not None and (
        expanded.get("via") == "graph" or expanded.get("graph_score", 0) > 0
    )
    assert expanded and expanded.get("graph_score", 0) > 0

    assert make_store(tmp_path / "empty").search(
        "anything", limit=3, fusion="rrf", graph_expand=True
    )["retrieval"].get("graph_expanded", 0) == 0


def test_temporal_retrieval(tmp_path):
    store = make_store(tmp_path)
    for text in ("Deployment rollback procedure for the API gateway",
                 "Gateway latency spike investigation notes",
                 "Unrelated billing reconciliation notes"):
        store.store(text, project="p")

    dated = store.search("gateway issues yesterday", limit=5, project="p",
                         fusion="rrf", temporal=True)
    assert dated["retrieval"].get("temporal_window") is not None
    assert any(r.get("fts_score", 0) > 0 for r in dated["results"])
    assert "ateway" in dated["results"][0]["content"]

    plain = store.search("gateway issues", limit=5, project="p", fusion="rrf", temporal=True)
    assert plain["retrieval"].get("temporal_window") is None


def test_query_aware_boosts_and_lexical_anchor(tmp_path):
    store = make_store(tmp_path)
    exact = store.store(
        "Error ERR_4512 raised by module auth_service on login", project="p"
    )["memory_id"]
    linked = store.store(
        "Troubleshooting notes for authentication failures in production", project="p"
    )["memory_id"]
    store.link(exact, linked, "related_to")

    # Lookup-shaped query: FTS leads and the strongest BM25 hit is pinned at
    # rank 1 even though the linked graph candidate carries a graph bonus.
    lookup = store.search(
        "ERR_4512 auth_service", limit=5, project="p", fusion="rrf", graph_expand=True
    )
    assert lookup["results"][0]["id"] == exact
    assert lookup["results"][0].get("anchored") is True
    assert lookup["retrieval"].get("anchored") == exact
    assert linked in [r["id"] for r in lookup["results"]]

    # Conceptual query: no lexical anchor is applied.
    conceptual = store.search(
        "what do we know about authentication failures?",
        limit=5, project="p", fusion="rrf", graph_expand=True,
    )
    assert "anchored" not in conceptual["retrieval"]


def test_failure_containment(tmp_path):
    store = make_store(tmp_path)
    store.store("A memory about container scheduling", project="p")

    assert isinstance(store.search("container", limit=3, project="p", fusion="nonsense"), dict)
    assert isinstance(store.search("", limit=3, project="p", fusion="rrf",
                                   graph_expand=True, temporal=True), dict)
    assert isinstance(make_store(tmp_path / "empty").search(
        "anything", limit=3, fusion="rrf", graph_expand=True, temporal=True), dict)
    assert len(store.search("container scheduling", limit=3, project="p", fusion="rrf",
                            graph_expand=True, temporal=True)["results"]) >= 1

    store.store("Scoped memory in another project", project="other")
    scoped = store.search("memory", limit=10, project="other", fusion="rrf", graph_expand=True)
    assert all(r.get("project") in ("other", None) for r in scoped["results"])


# ── Core-layer extensions ─────────────────────────────────────────────────

def test_scope_and_session_filters(tmp_path):
    store = make_store(tmp_path)
    store.store("Global preference about tabs", scope="global")
    store.store("Project convention about poetry", scope="project", project="repo-a")
    store.store("Session detail about oauth", scope="session", session_id="s1")

    assert store.search("preference", scope="global")["results"]
    assert store.search("convention", scope="project", project="repo-a")["results"]
    assert store.search("oauth", scope="session")["results"]
    # Scope is a hard filter on every retrieval path, including the recency
    # backfill that seeds candidates for a query with few lexical matches.
    global_hits = store.search("convention", scope="global")["results"]
    assert all(hit["scope"] == "global" for hit in global_hits)
    assert all(hit["content"] != "Project convention about poetry" for hit in global_hits)
    project_hits = store.search("oauth", scope="project")["results"]
    assert all(hit["scope"] == "project" for hit in project_hits)
    assert all(hit["content"] != "Session detail about oauth" for hit in project_hits)


def test_degenerate_content_rejected(tmp_path):
    store = make_store(tmp_path)
    for bad in ("", "   ", "...", "n/a"):
        result = store.store(bad)
        assert result["success"] is False
    assert store.health()["total_memories"] == 0


def test_fts_self_heal(tmp_path):
    path = str(tmp_path / "memories.db")
    store = MemoryStore(path)
    store.store("The parser handles nested expressions", project="p")
    assert store.search("parser", project="p")["results"]

    # Corrupt the index by wiping it directly, then reopen.
    store.db.execute("DELETE FROM memories_fts")
    store.db.commit()
    store.close()

    reopened = MemoryStore(path)
    assert reopened.health()["fts_ok"] is True
    assert reopened.search("parser", project="p")["results"]


def test_health_fields(tmp_path):
    store = make_store(tmp_path)
    first = store.store("Alpha memory about gateways", project="p")["memory_id"]
    second = store.store("Beta memory about gateways", project="p")["memory_id"]
    store.link(first, second, "related_to")
    store.supersede(first, "Alpha memory about gateways v2", project="p")

    health = store.health()
    for field in ("total_memories", "active_memories", "total_links", "total_conflicts",
                  "total_superseded", "duplicate_content_groups", "oldest_memory",
                  "newest_memory", "memories_by_type", "memories_by_project",
                  "fts_ok", "db_size_bytes", "schema_version"):
        assert field in health
    assert health["total_superseded"] >= 1
    assert health["fts_ok"] is True


def test_unlink(tmp_path):
    store = make_store(tmp_path)
    first = store.store("Alpha memory")["memory_id"]
    second = store.store("Beta memory")["memory_id"]
    assert store.link(first, second, "related_to")["success"]
    assert store.related(first)
    assert store.unlink(first, second)["success"]
    assert store.related(first) == []


def test_concurrent_store_and_search(tmp_path):
    store = make_store(tmp_path)
    errors = []

    def writer():
        try:
            for i in range(10):
                store.store(f"Concurrent fact number {i} about threading", project="p")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    for i in range(10):
        store.search(f"concurrent fact {i}", limit=3, project="p")
    thread.join()

    assert not errors
    assert store.health()["total_memories"] == 10
