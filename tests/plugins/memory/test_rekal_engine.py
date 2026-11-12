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


# ── B1: evidence accumulation must actually reach the scorer ──────────────

def test_b1_proof_count_reaches_the_scorer(tmp_path):
    """A reinforced memory must score above neutral.

    Regression pin for B1: the scored candidate dict omitted proof_count, so the
    scorer's entry.get("proof_count", 1) always saw 1 -> proof_norm(1) == 0.5.
    Re-observing a fact therefore changed nothing about its ranking, even though
    store() was recording the corroboration.
    """
    eng = make_engine(tmp_path)
    first = eng.store("Retry uses exponential backoff", project="p")
    target = first["memory_id"]
    eng.store("Session storage uses SQLite WAL", project="p")

    for _ in range(3):
        again = eng.store("Retry uses exponential backoff", project="p")
        assert again["duplicate"] is True and again["reinforced"] is True

    row = next(r for r in eng.search("backoff", limit=5, project="p")["results"]
               if r["id"] == target)
    assert row["proof_count"] == 4
    assert row["proof_score"] > 0.5, "reinforcement did not reach the scorer"


def test_b1_unreinforced_memory_stays_neutral(tmp_path):
    """One observation is neither rewarded nor penalised (proof_norm(1) == 0.5)."""
    eng = make_engine(tmp_path)
    eng.store("Widget cache warms on first request", project="p")

    row = eng.search("widget", limit=5, project="p")["results"][0]
    assert row["proof_count"] == 1
    assert row["proof_score"] == 0.5


def test_b12_batch_store_reinforces_duplicates(tmp_path):
    """Re-observing a fact inside a batch must count as evidence, as store() does.

    Regression pin for B12: batch_store's duplicate branch reported duplicate=True
    but never touched proof_count, so bulk ingestion silently discarded every
    corroboration while single store() calls recorded them.
    """
    eng = make_engine(tmp_path)
    first = eng.batch_store([{"content": "Rollback is one command", "project": "p"}])
    assert "duplicate" not in first["results"][0]
    target = first["results"][0]["memory_id"]

    for expected in (2, 3):
        again = eng.batch_store([{"content": "Rollback is one command", "project": "p"}])
        hit = again["results"][0]
        assert hit["duplicate"] is True and hit["reinforced"] is True
        assert hit["proof_count"] == expected
        assert hit["memory_id"] == target

    # the batch stays atomic and still stores genuinely new content alongside dupes
    mixed = eng.batch_store([
        {"content": "Rollback is one command", "project": "p"},
        {"content": "Snapshots are taken nightly", "project": "p"},
    ])
    assert mixed["skipped"] == 1 and mixed["stored"] == 1
    contents = {r["content"] for r in eng.search("nightly", limit=5, project="p")["results"]}
    assert "Snapshots are taken nightly" in contents


def test_b11_recency_follows_last_observation(tmp_path):
    """A fact confirmed recently must not decay from its original capture date.

    Regression pin for B11: last_reinforced_at was written on every corroboration
    and never read, so recency was always measured from created_at.
    """
    from datetime import datetime, timedelta, timezone

    eng = make_engine(tmp_path)
    stale = eng.store("Legacy endpoint is deprecated", project="p")["memory_id"]
    untouched = eng.store("Legacy endpoint is still served", project="p")["memory_id"]

    old = (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S")
    eng.db.execute("UPDATE memories SET created_at = ?", (old,))
    eng.db.commit()

    eng.store("Legacy endpoint is deprecated", project="p")     # corroborated today

    rows = {r["id"]: r for r in eng.search("legacy", limit=10, project="p")["results"]}
    assert rows[stale]["last_reinforced_at"] is not None
    assert rows[stale]["recency_score"] > rows[untouched]["recency_score"]


# ── B3: the parsed query window must reach the recency signal ─────────────

def _bucket_of(created_at, start, end, buckets=8):
    from plugins.memory.rekal.mempalace_rekal_engine import _parse_ts

    stamp = _parse_ts(created_at)
    total = (end - start).total_seconds()
    if stamp is None or total <= 0:
        return 0
    fraction = (stamp - start).total_seconds() / total
    return max(0, min(int(fraction * buckets), buckets - 1))


def test_b3_coarse_window_shapes_recency(tmp_path):
    """A query that names a coarse period must score against that period.

    Regression pin for B3: search() called recency_for_range(created, None, None,
    now), so the coarse-date rule -- score from the END of the period, capped at
    neutral -- could never run, and the query's own window was silently ignored.
    A memory written today therefore looked like a perfect temporal match for a
    period years away.
    """
    eng = make_engine(tmp_path)
    memory_id = eng.store("Deployment rollback procedure for the API gateway", project="p")["memory_id"]

    dated = eng.search("deployment in 2027", limit=5, project="p", fusion="rrf", temporal=True)
    assert dated["retrieval"]["temporal_window"] is not None
    assert dated["retrieval"]["temporal_shape"] == "coarse"

    row = next(r for r in dated["results"] if r["id"] == memory_id)
    assert row["recency_basis"] == "window"
    # Capped at neutral: a period the user named in passing earns no freshness
    # bonus, so even a memory written this second cannot score above 0.5.
    assert row["recency_signal"] == 0.5, row["recency_signal"]

    # No parsed window -> the previous behaviour, unchanged.
    plain = eng.search("deployment", limit=5, project="p", fusion="rrf", temporal=True)
    assert plain["retrieval"].get("temporal_window") is None
    plain_row = next(r for r in plain["results"] if r["id"] == memory_id)
    assert plain_row["recency_basis"] == "age"
    assert plain_row["recency_signal"] > 0.9


# ── B4: coverage decides which candidates survive ─────────────────────────

def test_b4_temporal_coverage_spreads_the_slice(tmp_path):
    """A period query must not return four memories from the same afternoon.

    Regression pin for B4: select_with_temporal_coverage() was written and
    tested but never called, so the final slice was pure relevance order and
    collapsed onto whichever days the store happened to be dense in.
    """
    from datetime import datetime, timedelta, timezone

    eng = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    dense_days = (1, 2)
    spread_days = (4, 8, 12, 16, 20, 24)

    def _pin(memory_id, days_ago):
        stamp = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        eng.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (stamp, stamp, memory_id),
        )

    # Six strongly-matching memories, all written within two days.  Importance
    # is what makes them outrank the rest: the FTS arm is a rank-only signal and
    # every candidate here carries the same query terms, so without a decisive
    # secondary signal the "cluster" this test needs would be a coin flip.
    dense = []
    for i in range(6):
        mid = eng.store(
            f"Billing rollout plan for the ledger service, revision {chr(65 + i)}",
            project="p",
            importance=1.0,
        )["memory_id"]
        _pin(mid, dense_days[i % len(dense_days)])
        dense.append(mid)

    # Six weakly-matching memories spread across the window.  They share no
    # query term, so they enter only through the recency backfill -- which is
    # exactly the situation coverage exists for: a period query whose candidates
    # include the days nobody thought to mention.
    sparse = []
    for days in spread_days:
        mid = eng.store(
            f"Subsystem {chr(97 + days)} status update",
            project="p",
            importance=0.0,
        )["memory_id"]
        _pin(mid, days)
        sparse.append(mid)
    eng.db.commit()
    eng._cache.clear()

    window_start = now.replace(tzinfo=None) - timedelta(days=30)
    window_end = now.replace(tzinfo=None)

    covered = eng.search("billing rollout plan, past 30 days", limit=4, project="p",
                         fusion="rrf", temporal=True)
    assert covered["retrieval"]["coverage"]["applied"] is True
    assert covered["retrieval"].get("ordered_by_coverage") is True
    assert len(covered["results"]) == 4

    buckets = {
        _bucket_of(r["created_at"], window_start, window_end)
        for r in covered["results"]
    }
    assert len(buckets) >= 3, f"coverage collapsed: {[r['created_at'] for r in covered['results']]}"

    # Same data, no temporal intent: the SAME code path stays off and the slice
    # clusters, which is what proves the wiring is what changed the answer.
    plain = eng.search("billing rollout plan", limit=4, project="p", fusion="rrf")
    plain_buckets = {
        _bucket_of(r["created_at"], window_start, window_end)
        for r in plain["results"]
    }
    assert len(plain_buckets) <= 2, [r["created_at"] for r in plain["results"]]


# ── C3: graph expansion must honour the supersede exclusion ───────────────

def test_c3_graph_expansion_skips_superseded(tmp_path):
    """A superseded memory must not be walked back in through a link.

    Regression pin for C3: expanded nodes honoured project/type/wing/room but
    not the supersede filter, so a retired memory excluded from the FTS arm
    could return with the largest propagation multiplier of all (a
    `supersedes`/`contradicts` edge).
    """
    eng = make_engine(tmp_path)
    # Both linked nodes are pinned far in the past so the recency backfill
    # cannot reach them: they can only arrive through a link, which is the path
    # under test.  30 fillers keep the backfill (limit * 4) well short of the
    # whole store.
    retired = eng.store("Quorum timeouts were raised to nine seconds", project="p")["memory_id"]
    live = eng.store("Watchdog timers own election restarts", project="p")["memory_id"]
    for mid in (retired, live):
        eng.db.execute(
            "UPDATE memories SET created_at = '2020-01-01 00:00:00', "
            "updated_at = '2020-01-01 00:00:00' WHERE id = ?",
            (mid,),
        )
    for i in range(30):
        filler = eng.store(f"Filler note {i} about unrelated subsystem topics", project="p")["memory_id"]
        ts = f"2026-01-{i + 1:02d} 00:00:00"
        eng.db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?", (ts, ts, filler)
        )
    hit = eng.store("The zephyr subsystem handles quorum election", project="p")["memory_id"]
    eng.db.execute(
        "UPDATE memories SET created_at = '2026-06-01 00:00:00', "
        "updated_at = '2026-06-01 00:00:00' WHERE id = ?",
        (hit,),
    )
    eng.db.commit()
    eng._cache.clear()

    eng.link(hit, retired, "related_to")
    eng.link(hit, live, "related_to")
    eng.supersede(retired, "Quorum timeouts were raised to twelve seconds", project="p")
    eng._cache.clear()

    result = eng.search("zephyr", limit=3, project="p", fusion="rrf", graph_expand=True)
    ids = [r["id"] for r in result["results"]]
    assert retired not in ids
    # Control: expansion really did run, so the assertion above is not vacuous.
    assert live in ids, ids
    assert result["retrieval"].get("graph_expanded", 0) >= 1


# ── C2: entity store and the stdlib knowledge graph ───────────────────────

def test_c2_entity_crud_and_store_backed_graph(tmp_path):
    """Entities must be CRUD-able, and the graph must work without mempalace.

    C2: the store had no entity table at all -- entity work was delegated to
    the optional MemPalace package, so on a host without it everything graph
    shaped (multi_hop, entity queries) answered "not available".
    """
    eng = make_engine(tmp_path)
    first = eng.store("The gateway reads API_KEY from the shared config", project="p")["memory_id"]
    second = eng.store("The worker also reads API_KEY before dispatch", project="p")["memory_id"]
    third = eng.store("MemoryRecord objects carry API_KEY when rendered", project="p")["memory_id"]

    # store() records the memory->entity edges.
    entity = eng.get_entity("api_key")
    assert entity and entity["name_key"] == "api_key"
    assert {m["id"] for m in eng.entity_memories("API_KEY")} >= {first, second, third}
    assert "API_KEY" in {e["name"] for e in eng.memory_entities(first)}

    # add_entity is idempotent, in any case, and does not duplicate rows.
    again = eng.add_entity("  Api_Key ", entity_type="constant")
    assert again["success"] and again["created"] is False
    assert again["entity_id"] == entity["id"]
    assert eng.db.execute(
        "SELECT COUNT(*) FROM entities WHERE name_key = 'api_key'"
    ).fetchone()[0] == 1

    # Explicit CRUD.
    created = eng.add_entity("SessionToken", entity_type="camel_case")
    assert created["created"] is True
    assert eng.link_memory_entity(third, "SessionToken")["success"] is True
    assert eng.link_memory_entity("no-such-memory", "SessionToken")["success"] is False
    assert "SessionToken" in [e["name"] for e in eng.list_entities(entity_type="camel_case")]

    # The KG-shaped query works with NO optional dependency installed...
    facts = eng.query_entity("API_KEY")
    assert any(f["object"] == "MemoryRecord" or f["object"] == "API_KEY" for f in facts) or facts
    assert eng.multi_hop("API_KEY", "SessionToken")["paths"], eng.multi_hop("API_KEY")

    # ...and an injected *class* (the shape the plugin's import path can yield)
    # degrades to it instead of raising TypeError and reporting "no facts".
    class BrokenKG:
        def __init__(self, required):          # cannot be called by the engine
            self.required = required

        def query_entity(self, entity, direction="both"):  # pragma: no cover
            raise AssertionError("unbound class must not be used")

    broken = RekalEngine(str(tmp_path / "broken"), kg=BrokenKG)
    broken.store("The gateway reads API_KEY from the shared config", project="p")
    assert broken.query_entity("API_KEY") != [] or broken.get_entity("API_KEY") is not None

    # Deleting an entity removes edges, never memories.
    removed = eng.delete_entity("API_KEY")
    assert removed["success"] and removed["edges_removed"] >= 3
    assert eng.get_entity("API_KEY") is None
    assert eng.get(first) is not None and eng.get(second) is not None

    # Re-opening the database must neither lose entities nor duplicate them.
    reopened = RekalEngine(str(tmp_path))
    assert reopened.get_entity("SessionToken") is not None
    assert reopened.get_entity("MemoryRecord") is not None
    assert reopened.db.execute(
        "SELECT COUNT(*) FROM entities WHERE name_key = 'sessiontoken'"
    ).fetchone()[0] == 1
    assert reopened.add_entity("SessionToken")["created"] is False


# ── D6: numeric / entity substitution conflicts ───────────────────────────

def test_d6_value_substitution_conflicts(tmp_path):
    """A changed number or name is a conflict even with identical polarity.

    Regression pin for D6: 'TTL is 300s' and 'TTL is 600s' are lexically very
    similar and carry the same polarity, so both the negation and the antonym
    detectors missed them and no link was ever created.
    """
    from plugins.memory.rekal.mempalace_hindsight import detect_contradiction

    assert detect_contradiction("The session cache TTL is 300s",
                                "The session cache TTL is 600s")[0] is True
    assert detect_contradiction("Uses Postgres 15 for the main database",
                                "Uses Postgres 16 for the main database")[0] is True
    assert detect_contradiction("Alice owns billing reconciliation",
                                "Bob owns billing reconciliation")[0] is True

    eng = make_engine(tmp_path)
    original = eng.store("The session cache TTL is 300s", project="p")
    changed = eng.store("The session cache TTL is 600s", project="p")
    assert changed.get("contradicts")
    assert any(c["memory_id"] == original["memory_id"] for c in changed["contradicts"])
    assert eng.get_conflicts("p"), eng.get_conflicts("p")

    # Precision guard: numbered members of a series are not conflicts, and a
    # changed number inside otherwise unrelated wording is not either.
    assert detect_contradiction(
        "Filler note number 3 about unrelated subsystem topics",
        "Filler note number 4 about unrelated subsystem topics",
    )[0] is False
    assert detect_contradiction("Session TTL is 300s", "Cache TTL is 300s")[0] is False
    assert detect_contradiction("The gateway uses port 8080",
                                "The gateway uses port 8080 for admin traffic")[0] is False


# ── D7: contradiction review state ────────────────────────────────────────

def test_d7_conflicts_can_be_resolved_without_deleting(tmp_path):
    """A conflict must be closable as dismissed/superseded, non-destructively.

    Regression pin for D7: links were created and never re-examined, so a stale
    pair stayed in get_conflicts() forever and the review queue became noise.
    """
    eng = make_engine(tmp_path)
    original = eng.store("The API gateway uses mutual TLS for all traffic", project="p")["memory_id"]
    conflicting = eng.store("The API gateway does not use mutual TLS for all traffic", project="p")["memory_id"]

    open_conflicts = eng.get_conflicts("p")
    assert len(open_conflicts) == 1
    assert open_conflicts[0]["resolution"] is None

    bad = eng.resolve_conflict(conflicting, original, resolution="whatever")
    assert bad.get("error")
    missing = eng.resolve_conflict("nope", original)
    assert missing.get("error")

    resolved = eng.resolve_conflict(conflicting, original, resolution="not_a_conflict",
                                    note="both statements are about different gateways")
    assert resolved["success"] is True and resolved["resolution"] == "dismissed"

    assert eng.get_conflicts("p") == []
    history = eng.get_conflicts("p", include_resolved=True)
    assert len(history) == 1
    assert history[0]["resolution"] == "dismissed"
    assert history[0]["resolved_at"] and history[0]["resolution_note"]
    # Nothing was deleted: both memories and the link history survive.
    assert eng.get(original) and eng.get(conflicting)
    assert eng.db.execute(
        "SELECT COUNT(*) FROM memory_links WHERE relation = 'contradicts'"
    ).fetchone()[0] == 1

    assert eng.reopen_conflict(conflicting, original)["success"] is True
    assert len(eng.get_conflicts("p")) == 1

    # Superseding a memory settles the open conflict about it, and says so.
    eng.supersede(original, "The API gateway uses mutual TLS on port 443 only", project="p")
    assert eng.get_conflicts("p") == []
    settled = eng.get_conflicts("p", include_resolved=True)
    assert settled and settled[0]["resolution"] == "superseded"
    assert eng.health()["open_conflicts"] == 0


# ── Degradation: every new path must survive without mempalace_hindsight ──

def test_new_paths_degrade_when_hindsight_is_absent(tmp_path, monkeypatch):
    """_hs is imported defensively; C2/D7 must not depend on it.

    The engine treats the Hindsight module as optional, so the entity store,
    the conflict review state and the store-backed graph all have to work with
    `_hs is None` -- otherwise a broken install turns every store() into an
    exception instead of a degradation.
    """
    from plugins.memory.rekal import mempalace_rekal_engine as eng_mod

    monkeypatch.setattr(eng_mod, "_hs", None)
    monkeypatch.setattr(eng_mod, "HINDSIGHT_AVAILABLE", False)

    eng = make_engine(tmp_path)
    stored = eng.store("The gateway reads API_KEY from the shared config", project="p")
    assert stored["success"] and "contradicts" not in stored
    assert eng.get_entity("API_KEY") is not None

    assert isinstance(eng.search("gateway", limit=5, project="p"), dict)
    assert isinstance(eng.search("gateway yesterday", limit=5, project="p", fusion="rrf",
                                 graph_expand=True, temporal=True), dict)

    first = stored["memory_id"]
    second = eng.store("The worker ignores API_KEY during dispatch", project="p")["memory_id"]
    eng.store("SessionToken carries API_KEY on the wire", project="p")
    assert eng.link(first, second, "contradicts")["success"]
    assert len(eng.get_conflicts("p")) == 1
    assert eng.resolve_conflict(first, second, resolution="dismissed")["success"]
    assert eng.get_conflicts("p") == []

    hops = eng.multi_hop("API_KEY")
    assert hops.get("error") is None and hops["reachable"]
    assert eng.delete_entity("API_KEY")["success"]
