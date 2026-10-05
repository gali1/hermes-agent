"""Integration tests for the upstream Rekal memory engine in Hermes.

Ported from OpenCode's ``mempalace_rekal_test.py`` (baseline non-regression,
migration, evidence accumulation, contradiction detection, rank fusion, graph
expansion, temporal retrieval, failure containment) plus a Hermes-specific
test for migrating the schema written by the pre-replacement plugin port.

These tests drive a real SQLite database in a temp directory.  The ChromaDB
vector side is exercised opportunistically: when ``mempalace`` is installed
the engine writes drawers as a side effect, so the palace path is redirected
to a throwaway directory BEFORE any mempalace import; when it is not the
engine degrades to FTS-only, which is itself one of the behaviours under test.
"""

import os
import sqlite3
import tempfile

# Redirect the MemPalace/ChromaDB side to a throwaway directory BEFORE the
# engine (or mempalace) is imported.  Without this the suite could write
# drawers into the developer's real palace.
_PALACE = tempfile.mkdtemp(prefix="rekal-test-palace-")
os.environ.setdefault("MEMPALACE_PALACE_PATH", _PALACE)
os.environ.setdefault("MEMPALACE_DATA_DIR", _PALACE)

from plugins.memory.rekal.mempalace_rekal_engine import RekalEngine, _quote_fts  # noqa: E402


def make_engine(tmp_path):
    return RekalEngine(str(tmp_path))


# ── Non-regression: the baseline contract ─────────────────────────────────

def test_baseline_unchanged(tmp_path):
    eng = make_engine(tmp_path)
    for text in ("Session storage uses SQLite WAL", "Retry uses exponential backoff"):
        eng.store(text, project="p")

    result = eng.search("storage", limit=5, project="p")
    assert sorted(result.keys()) == ["query", "results", "total_candidates", "weights"]
    assert "retrieval" not in result

    row = result["results"][0]
    for field in ("id", "content", "memory_type", "project", "wing", "room", "tags",
                  "importance", "created_at", "updated_at", "access_count",
                  "score", "fts_score", "vec_score", "recency_score", "access_score", "source"):
        assert field in row

    assert "rrf_score" not in row
    assert set(result["weights"]) >= {"w_fts", "w_vec", "w_recency", "half_life"}

    assert _quote_fts("alpha beta") == '"alpha"* "beta"*'
    assert _quote_fts("alpha beta", match_any=True) == '"alpha"* OR "beta"*'


def test_existing_operations_still_work(tmp_path):
    eng = make_engine(tmp_path)
    first = eng.store("Original content about caching", project="p")["memory_id"]
    second = eng.store("Second memory about indexing", project="p")["memory_id"]

    assert first
    assert eng.get(first)["content"] == "Original content about caching"
    assert eng.update(first, content="Updated content about caching")["success"]
    assert "Updated" in eng.get(first)["content"]
    assert eng.link(first, second, "related_to")["success"]
    assert len(eng.related(first)) >= 1

    superseded = eng.supersede(first, "Superseded content about caching", project="p")
    assert superseded.get("new_id") or superseded.get("memory_id")

    assert eng.health()["total_memories"] >= 2
    assert isinstance(eng.topics("p"), (dict, list))
    assert isinstance(eng.timeline("p"), (dict, list))
    assert eng.delete(second)["success"]
    assert eng.get(second) is None or eng.get(second) == {}


# ── Schema migration and backward compatibility ───────────────────────────

def test_migration_from_pre_upgrade_db(tmp_path):
    directory = str(tmp_path)
    path = os.path.join(directory, "rekal_memories.db")

    # Build a database with the OLD upstream schema: no proof_count, no
    # last_reinforced_at.
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT,
            memory_type TEXT NOT NULL DEFAULT 'fact', project TEXT, wing TEXT, room TEXT,
            tags TEXT, importance REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0, last_accessed_at TEXT);
        CREATE TABLE memory_links (
            from_id TEXT NOT NULL, to_id TEXT NOT NULL, relation TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY (from_id, to_id, relation));
        CREATE TABLE rekal_config (
            project TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY (project, key));
        INSERT INTO memories (id, content, content_hash, memory_type, created_at, updated_at)
        VALUES ('legacy1', 'A memory written before the upgrade', 'hash-legacy', 'fact',
                '2020-01-01 00:00:00', '2020-01-01 00:00:00');
        """
    )
    old.commit()
    old.close()

    eng = RekalEngine(directory)
    columns = {row[1] for row in eng.db.execute("PRAGMA table_info(memories)")}
    assert "proof_count" in columns
    assert "last_reinforced_at" in columns

    legacy = eng.get("legacy1")
    assert legacy is not None and "before the upgrade" in legacy["content"]
    assert legacy.get("proof_count") == 1

    assert isinstance(eng.search("upgrade", limit=5), dict)
    assert isinstance(eng.search("upgrade", limit=5, fusion="rrf"), dict)

    # Re-opening must be idempotent.
    reopened = RekalEngine(directory)
    assert reopened.get("legacy1") is not None


def test_migration_from_hermes_legacy_plugin_schema(tmp_path):
    """The replaced Hermes plugin used source_id/target_id links and a
    project-less config table; opening that file with the upstream engine
    must migrate it in place without losing rows."""
    directory = str(tmp_path)
    path = os.path.join(directory, "rekal_memories.db")

    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'fact',
            tags TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 1.0,
            access_count INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}',
            superseded_by TEXT,
            superseded INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL
        );
        CREATE TABLE memory_links (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL DEFAULT 'related_to',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(source_id, target_id, relation)
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE rekal_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);

        INSERT INTO memories (id, content, content_hash, memory_type, tags, importance,
                              superseded_by, superseded, created_at, updated_at, accessed_at)
        VALUES ('m1', 'The gateway uses mutual TLS for all traffic', 'h1', 'fact', '[]', 8.0,
                NULL, 0, '2026-01-01 00:00:00', '2026-01-01 00:00:00', '2026-01-01 00:00:00');
        INSERT INTO memories (id, content, content_hash, memory_type, tags, importance,
                              superseded_by, superseded, created_at, updated_at, accessed_at)
        VALUES ('m2', 'The old gateway policy is in effect', 'h2', 'fact', '[]', 5.0,
                'm3', 1, '2026-01-01 00:00:00', '2026-01-02 00:00:00', '2026-01-01 00:00:00');
        INSERT INTO memories (id, content, content_hash, memory_type, tags, importance,
                              superseded_by, superseded, created_at, updated_at, accessed_at)
        VALUES ('m3', 'The new gateway policy replaced the old one', 'h3', 'fact', '[]', 4.0,
                NULL, 0, '2026-01-02 00:00:00', '2026-01-02 00:00:00', '2026-01-02 00:00:00');

        INSERT INTO memory_links (id, source_id, target_id, relation, created_at)
        VALUES ('l1', 'm1', 'm3', 'related_to', '2026-01-01 00:00:00');
        INSERT INTO memory_links (id, source_id, target_id, relation, created_at)
        VALUES ('l2', 'm1', 'm3', 'supports', '2026-01-01 00:00:00');
        INSERT INTO rekal_config (key, value) VALUES ('w_fts', '0.4');
        """
    )
    old.commit()
    old.close()

    eng = RekalEngine(directory)

    # Legacy importance 1-10 rescaled to 0-1.
    assert abs(eng.get("m1")["importance"] - 0.8) < 1e-9

    # Legacy links survive; unsupported relations fold into related_to.
    related = eng.related("m1")
    assert any(r.get("id") == "m3" for r in related)

    # Legacy supersede state preserved as a supersedes link.
    supersede_links = eng.db.execute(
        "SELECT COUNT(*) FROM memory_links WHERE relation = 'supersedes' "
        "AND from_id = 'm3' AND to_id = 'm2'"
    ).fetchone()[0]
    assert supersede_links == 1

    # Superseded rows are excluded from search results.
    result = eng.search("gateway", limit=10)
    assert all(r["id"] != "m2" for r in result["results"])

    # Config table migrated to the project-keyed shape.
    assert eng.set_config("p", "w_fts", 0.4).get("success") is True

    # Re-opening is idempotent.
    reopened = RekalEngine(directory)
    assert reopened.get("m1") is not None


# ── Evidence accumulation ─────────────────────────────────────────────────

def test_proof_accumulation(tmp_path):
    eng = make_engine(tmp_path)
    text = "The deploy script requires an explicit region flag"

    first = eng.store(text, project="p")
    assert not first.get("duplicate")

    second = eng.store(text, project="p")
    assert second.get("duplicate") is True
    assert second["memory_id"] == first["memory_id"]
    assert second.get("proof_count") == 2

    third = eng.store(text, project="p")
    assert third.get("proof_count") == 3

    stored = eng.get(first["memory_id"])
    assert stored.get("proof_count") == 3
    assert stored.get("last_reinforced_at")
    assert eng.db.execute("SELECT COUNT(*) FROM memories WHERE content = ?", (text,)).fetchone()[0] == 1


# ── Contradiction detection ───────────────────────────────────────────────

def test_contradiction_links(tmp_path):
    eng = make_engine(tmp_path)
    original = eng.store("The API gateway uses mutual TLS for all traffic", project="p")
    assert not original.get("contradicts")

    conflicting = eng.store("The API gateway does not use mutual TLS for all traffic", project="p")
    assert conflicting.get("contradicts")
    assert any(c["memory_id"] == original["memory_id"] for c in conflicting.get("contradicts", []))
    assert all(0.0 < c["confidence"] <= 0.95 for c in conflicting.get("contradicts", []))

    links = eng.db.execute("SELECT COUNT(*) FROM memory_links WHERE relation='contradicts'").fetchone()[0]
    assert links >= 1

    conflicts = eng.get_conflicts("p")
    reported = conflicts.get("conflicts", []) if isinstance(conflicts, dict) else conflicts
    assert len(reported) >= 1, f"got {reported!r}"

    assert eng.get(original["memory_id"]) is not None and eng.get(conflicting["memory_id"]) is not None

    unrelated = eng.store("Billing reconciliation runs nightly at 02:00 UTC", project="p")
    assert not unrelated.get("contradicts")


# ── Advanced retrieval ────────────────────────────────────────────────────

def test_rank_fusion(tmp_path):
    eng = make_engine(tmp_path)
    for text in ("Session storage uses SQLite with WAL mode",
                 "Retry policy uses exponential backoff",
                 "Auth tokens expire after thirty days"):
        eng.store(text, project="p")

    fused = eng.search("storage", limit=5, project="p", fusion="rrf")
    assert "retrieval" in fused
    assert fused["retrieval"]["mode"] in ("rrf", "rrf-empty")
    assert len(fused["results"]) >= 1

    if fused["retrieval"]["mode"] == "rrf":
        top = fused["results"][0]
        assert "rrf_score" in top
        assert top.get("rrf_rank") == 1
        assert isinstance(top.get("source_ranks"), dict)
        assert all(fused["results"][i]["score"] >= fused["results"][i + 1]["score"]
                   for i in range(len(fused["results"]) - 1))


def test_graph_expansion(tmp_path):
    eng = make_engine(tmp_path)
    # The orphan is stored first so the engine's recency backfill does not
    # reach it; it must therefore arrive purely through the link.  Timestamps
    # are pinned because FTS-only retrieval completes all 30 stores inside one
    # clock second, which would make the backfill's created_at ordering (and
    # therefore this test) depend on wall-clock timing.
    orphan = eng.store("Quorum timeouts were raised to nine seconds", project="p")["memory_id"]
    eng.db.execute(
        "UPDATE memories SET created_at = '2020-01-01 00:00:00', updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        (orphan,),
    )
    for i in range(30):
        filler = eng.store(f"Filler note number {i} about unrelated subsystem topics", project="p")["memory_id"]
        ts = f"2026-01-{i + 1:02d} 00:00:00"
        eng.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (ts, ts, filler),
        )
    hit = eng.store("The zephyr subsystem handles quorum election", project="p")["memory_id"]
    eng.db.execute(
        "UPDATE memories SET created_at = '2026-06-01 00:00:00', updated_at = '2026-06-01 00:00:00' WHERE id = ?",
        (hit,),
    )
    eng.db.commit()
    eng._cache.clear()
    eng.link(hit, orphan, "related_to")

    without = eng.search("zephyr", limit=3, project="p", fusion="rrf")
    assert orphan not in [r["id"] for r in without["results"]]

    with_graph = eng.search("zephyr", limit=3, project="p", fusion="rrf", graph_expand=True)
    ids = [r["id"] for r in with_graph["results"]]
    assert with_graph["retrieval"].get("graph_expanded", 0) >= 1
    assert orphan in ids

    expanded = next((r for r in with_graph["results"] if r["id"] == orphan), None)
    assert expanded is not None and (expanded.get("via") == "graph" or expanded.get("graph_score", 0) > 0), \
        f"got via={expanded and expanded.get('via')} graph={expanded and expanded.get('graph_score')}"
    assert expanded and expanded.get("graph_score", 0) > 0

    assert make_engine(tmp_path / "empty").search(
        "anything", limit=3, fusion="rrf", graph_expand=True
    )["retrieval"].get("graph_expanded", 0) == 0


def test_temporal_retrieval(tmp_path):
    eng = make_engine(tmp_path)
    for text in ("Deployment rollback procedure for the API gateway",
                 "Gateway latency spike investigation notes",
                 "Unrelated billing reconciliation notes"):
        eng.store(text, project="p")

    dated = eng.search("gateway issues yesterday", limit=5, project="p", fusion="rrf", temporal=True)
    assert dated["retrieval"].get("temporal_window") is not None
    assert any(r.get("fts_score", 0) > 0 for r in dated["results"])
    assert "ateway" in dated["results"][0]["content"]

    plain = eng.search("gateway issues", limit=5, project="p", fusion="rrf", temporal=True)
    assert plain["retrieval"].get("temporal_window") is None


def test_advanced_retrieval_is_contained(tmp_path):
    eng = make_engine(tmp_path)
    eng.store("A memory about container scheduling", project="p")

    assert isinstance(eng.search("container", limit=3, project="p", fusion="nonsense"), dict)
    assert isinstance(eng.search("", limit=3, project="p", fusion="rrf",
                                 graph_expand=True, temporal=True), dict)
    assert isinstance(make_engine(tmp_path / "empty").search(
        "anything", limit=3, fusion="rrf", graph_expand=True, temporal=True), dict)
    assert len(eng.search("container scheduling", limit=3, project="p", fusion="rrf",
                          graph_expand=True, temporal=True)["results"]) >= 1

    eng.store("Scoped memory in another project", project="other")
    scoped = eng.search("memory", limit=10, project="other", fusion="rrf", graph_expand=True)
    assert all(r.get("project") in ("other", None) for r in scoped["results"])
