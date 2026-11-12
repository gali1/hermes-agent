"""B5 / B6 / D5 / D9: near-duplicate wiring, evidence attribution, quality
gate, non-destructive consolidation.

The B5 tests fail on the pre-change store because the insert path is exact-hash
only; the B6 tests fail because no evidence attribution is attached; the D9
test fails because there is no ``consolidate`` method.
"""

import agent.memory.store as store_module
from agent.memory.dedup import find_near_duplicate, is_duplicate
from agent.memory.store import MemoryStore

# 0.9813 trigram similarity: same fact, different content_hash.  The exact-hash
# path must miss it and the fuzzy path must catch it.
BASE = (
    "The deployment pipeline requires an explicit region flag for every "
    "production rollout and refuses to continue without one"
)
NEAR = (
    "The deployment pipeline requires an explicit region flag for every "
    "production rollouts and refuses to continue without one"
)


def make_store(tmp_path, **kwargs):
    return MemoryStore(str(tmp_path / "memories.db"), **kwargs)


# ── B5: fuzzy near-duplicate on the insert path ───────────────────────────

def test_near_duplicate_is_reinforced_not_inserted(tmp_path):
    assert is_duplicate(BASE, NEAR)          # premise: fuzzy match at >= 0.97
    assert BASE != NEAR                      # premise: content_hash differs

    store = make_store(tmp_path)
    first = store.store(BASE, project="p")
    second = store.store(NEAR, project="p")

    assert second["duplicate"] is True
    assert second.get("fuzzy") is True
    assert second["memory_id"] == first["memory_id"]
    assert second["proof_count"] == 2
    assert second["similarity"] >= 0.97
    # Reinforced, never merged or deleted: one row, original text intact.
    assert store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    row = store.get(first["memory_id"])
    assert row["content"] == BASE
    assert row["proof_count"] == 2


def test_fuzzy_hit_raises_confidence_but_never_deletes(tmp_path):
    store = make_store(tmp_path)
    first = store.store(BASE, project="p", confidence=0.5)
    second = store.store(NEAR, project="p")

    assert second["confidence"] > 0.5
    row = store.get(first["memory_id"])
    assert row["confidence"] == second["confidence"]
    assert row["content"] == BASE
    assert store.health()["total_memories"] == 1


def test_exact_hash_fast_path_skips_the_fuzzy_scan(tmp_path, monkeypatch):
    calls = []
    real = store_module.find_near_duplicate

    def _counting(text, candidates, threshold=0.97):
        calls.append(text)
        return real(text, candidates, threshold)

    monkeypatch.setattr(store_module, "find_near_duplicate", _counting)

    store = make_store(tmp_path)
    store.store(BASE, project="p")           # hash miss -> one fuzzy scan
    assert len(calls) == 1
    store.store(BASE, project="p")           # exact hit -> no fuzzy scan
    assert len(calls) == 1
    store.store(BASE + " ", project="p")     # normalises to an exact hit too
    assert len(calls) == 1

    # The added cost applies only to a genuine miss.
    store.store("A completely different fact about billing ledgers", project="p")
    assert len(calls) == 2


def test_fuzzy_scan_is_bounded_and_project_scoped(tmp_path):
    def _seed(path, limit):
        store = MemoryStore(str(path), fuzzy_dedup_limit=limit)
        first = store.store(BASE, project="p")["memory_id"]
        for i in range(3):
            store.store(f"Unrelated recent note number {i} about queues", project="p")
        return store, first

    # Limit 2: only the two most recent rows are inspected, so the older
    # near-duplicate is out of range and the write goes through as new.
    bounded, _ = _seed(tmp_path / "bounded.db", 2)
    assert bounded.store(NEAR, project="p").get("duplicate") is not True
    assert bounded.health()["total_memories"] == 5

    # Default window sees it.
    wide, first = _seed(tmp_path / "wide.db", 50)
    hit = wide.store(NEAR, project="p")
    assert hit["duplicate"] is True and hit["memory_id"] == first
    assert wide.health()["total_memories"] == 4

    # A different project is never a fuzzy candidate.
    other = wide.store(NEAR, project="other")
    assert other.get("duplicate") is not True

    # Disabling the scan restores plain insert behaviour for content the store
    # has never seen (a third variant, not the already-reinforced one).
    assert wide.set_fuzzy_dedup(0)["success"] is True
    assert wide.store(NEAR + " (verified twice)", project="p").get("duplicate") is not True


def test_allow_duplicate_bypasses_fuzzy_path(tmp_path):
    store = make_store(tmp_path)
    store.store(BASE, project="p")
    forced = store.store(NEAR, project="p", allow_duplicate=True)
    assert forced.get("duplicate") is not True
    assert store.health()["total_memories"] == 2


# ── B6: evidence attribution only ─────────────────────────────────────────

def test_results_carry_evidence_attribution(tmp_path):
    store = make_store(tmp_path)
    user_memory = store.store(
        "Alpha gateway terminates mTLS with rotating certificates",
        project="p", source={"type": "user"},
    )["memory_id"]
    inferred = store.store(
        "Billing reconciliation runs nightly at 02:00 UTC",
        project="p", source={"type": "inferred"},
    )["memory_id"]

    result = store.search("alpha gateway billing", limit=10, project="p", track_access=False)
    by_id = {row["id"]: row for row in result["results"]}
    assert user_memory in by_id and inferred in by_id

    assert by_id[user_memory]["source_quality"] == 0.9
    assert by_id[inferred]["source_quality"] == 0.3
    assert 0.0 <= by_id[user_memory]["evidence_confidence"] <= 0.95
    assert by_id[user_memory]["evidence_confidence"] > by_id[inferred]["evidence_confidence"]
    # The stored per-record confidence is untouched by attribution.
    assert by_id[user_memory]["confidence"] == 0.5


def test_evidence_attribution_never_feeds_scoring(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.store("Alpha gateway terminates mTLS with rotating certificates",
                project="p", source={"type": "user"})
    store.store("Billing reconciliation runs nightly at 02:00 UTC",
                project="p", source={"type": "inferred"})

    before = store.search("alpha gateway billing", limit=10, project="p",
                          fusion="rrf", track_access=False)
    baseline = {row["id"]: row["score"] for row in before["results"]}
    assert before["results"][0]["evidence_confidence"] > 0

    # If confidence were a scoring input, this absurd value would reorder
    # everything.  Scores must be bit-identical.
    monkeypatch.setattr(store_module, "evidence_confidence", lambda *a, **k: 5.0)
    after = store.search("alpha gateway billing", limit=10, project="p",
                         fusion="rrf", track_access=False)
    assert {row["id"]: row["score"] for row in after["results"]} == baseline
    assert after["results"][0]["evidence_confidence"] == 5.0


# ── D5: degenerate quality gate ───────────────────────────────────────────

def test_quality_gate_rejects_degenerate_content_at_every_write(tmp_path):
    store = make_store(tmp_path)
    for bad in ("", "   ", "...", "n/a", "None", "null", "- -"):
        assert store.store(bad)["success"] is False

    real = store.store("A real memory about parser recovery")["memory_id"]
    assert store.update(real, content="...")["success"] is False
    assert store.supersede(real, "   ")["success"] is False
    assert store.health()["total_memories"] == 1


def test_near_duplicate_matcher_skips_degenerate_candidates(tmp_path):
    # The gate is dedup.is_degenerate; the shared matcher must apply it rather
    # than comparing meaningless text.
    assert find_near_duplicate("None", [("x", "none")]) is None
    assert find_near_duplicate("real content about queues", [("x", "...")]) is None
    assert find_near_duplicate("...", [("x", "real content about queues")]) is None
    assert find_near_duplicate("real content about queues",
                               [("x", "real content about queues")]) == ("x", 1.0)

    # A legacy degenerate row written before the gate existed can never be the
    # fuzzy target for a real write.
    store = make_store(tmp_path)
    store.db.execute(
        "INSERT INTO memories (id, content, content_hash, memory_type, created_at, updated_at) "
        "VALUES ('legacy-bad', '...', 'hash-bad', 'fact', ?, ?)",
        ("2020-01-01 00:00:00", "2020-01-01 00:00:00"),
    )
    store.db.commit()
    result = store.store("A fact that must not match punctuation", project=None)
    assert result.get("duplicate") is not True


# ── D9: non-destructive consolidation ─────────────────────────────────────

def test_consolidate_links_duplicates_without_deleting(tmp_path):
    store = make_store(tmp_path)
    canonical = store.store(BASE, project="p")["memory_id"]
    duplicate = store.store(NEAR, project="p", allow_duplicate=True)["memory_id"]
    unrelated = store.store("Nightly billing reconciliation owns the ledger", project="p")["memory_id"]
    assert store.health()["total_memories"] == 3

    report = store.consolidate(project="p")
    assert report["success"] is True
    assert report["deleted"] == 0
    assert report["groups_count"] == 1
    assert report["linked"] == [{
        "duplicate_id": duplicate,
        "canonical_id": canonical,
        "similarity": report["linked"][0]["similarity"],
        "proof_count": 1,
    }]

    # Links exist in both directions of meaning; nothing is removed.
    relations = {
        (row["from_id"], row["to_id"], row["relation"])
        for row in store.db.execute("SELECT from_id, to_id, relation FROM memory_links")
    }
    assert (duplicate, canonical, "duplicate_of") in relations
    assert (canonical, duplicate, "supersedes") in relations

    assert store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3
    original = store.get(duplicate)
    assert original is not None
    assert original["content"] == NEAR          # original content preserved
    assert original["proof_count"] == 1         # its own evidence preserved
    assert original["superseded_by"] == canonical
    assert original["status"] == "superseded"
    # Evidence is additive on the canonical row, not moved.
    assert store.get(canonical)["proof_count"] == 2

    # Canonical still retrievable, duplicate excluded from active retrieval.
    hits = store.search("deployment pipeline region flag", limit=10, project="p")
    ids = {row["id"] for row in hits["results"]}
    assert canonical in ids and duplicate not in ids
    assert unrelated in ids

    # Idempotent: a second pass has nothing active left to link.
    again = store.consolidate(project="p")
    assert again["linked"] == []
    assert store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3


def test_consolidate_dry_run_writes_nothing(tmp_path):
    store = make_store(tmp_path)
    store.store(BASE, project="p")
    store.store(NEAR, project="p", allow_duplicate=True)

    report = store.consolidate(project="p", dry_run=True)
    assert report["dry_run"] is True
    assert len(report["linked"]) == 1
    assert store.db.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0] == 0
    assert store.diagnostics()["orphan_links"]["count"] == 0
    assert {row["status"] for row in store.db.execute("SELECT status FROM memories")} == {"active"}
