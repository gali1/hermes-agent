"""Unit tests for the Hindsight-derived memory primitives.

Pure-function tests: no database, no network, no embedding backend. Ported
from OpenCode's ``mempalace_hindsight_test.py`` so the upstream contract is
pinned in Hermes too.
"""

import math
from datetime import datetime, timedelta

from plugins.memory.rekal.mempalace_hindsight import (
    BOOST_LEVELS,
    DEFAULT_RRF_K,
    analyze_query,
    boosted_rrf_score,
    combined_score,
    compute_recency_decay,
    detect_contradiction,
    expand_links,
    extract_temporal_constraint,
    is_degenerate,
    link_activation,
    proof_norm,
    reciprocal_rank_fusion,
    recency_for_range,
    select_with_temporal_coverage,
    spans_calendar_period,
    temporal_proximity,
    trigram_similarity,
)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def test_rrf():
    fused = reciprocal_rank_fusion([("fts", ["a", "b"]), ("vec", ["b", "a"])])
    assert sorted(r["id"] for r in fused) == ["a", "b"]
    assert close(fused[0]["rrf_score"], 1 / 61 + 1 / 62)

    # b is rank1 in vec and rank2 in fts; a is rank1 in fts and rank2 in vec.
    # Identical totals -> stable order, first appearance ("a") wins.
    assert fused[0]["id"] == "a"

    fused = reciprocal_rank_fusion([("fts", ["x", "y"]), ("vec", ["y"])])
    assert fused[0]["id"] == "y"
    assert [r["rrf_rank"] for r in fused] == [1, 2]
    assert fused[0]["source_ranks"] == {"fts_rank": 2, "vec_rank": 1}

    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([("fts", [])]) == []

    many = reciprocal_rank_fusion([("fts", [str(i) for i in range(50)])])
    assert [r["id"] for r in many] == [str(i) for i in range(50)]


def test_rrf_boosts():
    ranks = {"graph_rank": 10}
    base = 1.0 / (DEFAULT_RRF_K + 10)

    assert close(boosted_rrf_score(base, ranks, {}), base)
    assert close(boosted_rrf_score(base, ranks, {"vec": "high"}), base)
    assert close(boosted_rrf_score(base, ranks, {"graph": "bogus"}), base)

    low = boosted_rrf_score(base, ranks, {"graph": "low"})
    medium = boosted_rrf_score(base, ranks, {"graph": "medium"})
    high = boosted_rrf_score(base, ranks, {"graph": "high"})
    assert base < low < medium < high

    divisor = BOOST_LEVELS["medium"]
    boosted = boosted_rrf_score(1.0 / (DEFAULT_RRF_K + 30), {"graph_rank": 30}, {"graph": "medium"})
    rival = 1.0 / (DEFAULT_RRF_K + 10)  # 30 < 4*10, so boosted should win
    assert boosted > rival
    rival_far = 1.0 / (DEFAULT_RRF_K + 6)  # 30 > 4*6, so boosted should lose
    assert boosted < rival_far
    assert divisor == 4.0


# ── Recency ───────────────────────────────────────────────────────────────

def test_recency():
    assert close(compute_recency_decay(0), 1.0)
    assert close(compute_recency_decay(182.5, "linear", 365.0), 0.5)
    assert close(compute_recency_decay(10_000, "linear", 365.0), 0.1)
    assert close(compute_recency_decay(9999, "none"), 0.5)
    assert close(compute_recency_decay(90, "exponential", None, 90.0), 0.5)
    assert close(compute_recency_decay(-5, "exponential"), 1.0)
    assert all(0.0 <= compute_recency_decay(d) <= 1.0 for d in (0, 1, 100, 400, 100000))


def test_coarse_dates():
    now = datetime(2026, 8, 1)
    assert spans_calendar_period(datetime(2015, 1, 1), datetime(2015, 12, 31, 23, 59, 59))
    assert spans_calendar_period(datetime(2026, 3, 1), datetime(2026, 3, 31, 23, 59, 59))
    assert not spans_calendar_period(datetime(2026, 3, 1), datetime(2026, 3, 3))
    assert not spans_calendar_period(datetime(2026, 3, 5), datetime(2026, 3, 1))

    coarse = recency_for_range(datetime(2026, 1, 1), datetime(2026, 12, 31, 23, 59, 59), None, now)
    assert coarse <= 0.5

    precise = recency_for_range(datetime(2026, 7, 30), None, None, now)
    assert precise > 0.5
    assert close(recency_for_range(None, None, None, now), 0.5)
    assert close(recency_for_range(None, None, datetime(2026, 7, 30), now), precise)


# ── Evidence ──────────────────────────────────────────────────────────────

def test_proof_norm():
    assert close(proof_norm(1), 0.5)
    assert proof_norm(2) > 0.5
    assert close(proof_norm(3), 0.5 + math.log(3) / 10.0)
    assert proof_norm(2) < proof_norm(5) < proof_norm(20)
    assert proof_norm(10**9) <= 1.0
    assert close(proof_norm(0), 0.5) and close(proof_norm(-3), 0.5)
    assert close(proof_norm(None), 0.5)
    assert close(proof_norm("abc"), 0.5)


# ── Combined scoring ──────────────────────────────────────────────────────

def test_combined_score():
    assert close(combined_score(0.8), 0.8)
    assert combined_score(0.5, recency=1.0) > 0.5
    assert combined_score(0.5, recency=0.0) < 0.5

    best = combined_score(1.0, recency=1.0, importance=1.0, proof=1.0, graph=1.0)
    worst = combined_score(1.0, recency=0.0, importance=0.0, proof=0.0, graph=0.0)
    assert best < 1.26, f"got {best:.4f}"
    assert worst > 0.78, f"got {worst:.4f}"
    assert best / worst < 1.6, f"ratio={best / worst:.4f}"

    strong = combined_score(0.90, recency=0.0, importance=0.0, proof=0.0, graph=0.0)
    weak = combined_score(0.50, recency=1.0, importance=1.0, proof=1.0, graph=1.0)
    assert strong > weak, f"strong={strong:.4f} weak={weak:.4f}"

    assert close(combined_score(0.4, recency=1.0) / 0.4, combined_score(0.8, recency=1.0) / 0.8)


# ── Graph expansion ───────────────────────────────────────────────────────

def test_link_activation():
    assert close(link_activation(0), 0.0)
    assert close(link_activation(1), math.tanh(0.5))
    assert link_activation(1) < link_activation(2) < link_activation(3)
    assert link_activation(3) < 1.0
    assert link_activation(1000) <= 1.0
    assert close(link_activation(-5), 0.0)


def test_expand_links():
    adjacency = {
        "a": [("b", "related_to")],
        "b": [("c", "related_to")],
        "c": [("d", "related_to")],
    }
    out = expand_links(["a"], adjacency, max_hops=2)
    assert "a" not in out
    assert "b" in out
    assert "c" in out
    assert "d" not in out
    assert out["b"] > out["c"]
    assert close(out["b"], 0.7)

    strong = expand_links(["a"], {"a": [("b", "supersedes")]}, max_hops=1)
    weak = expand_links(["a"], {"a": [("b", "related_to")]}, max_hops=1)
    assert strong["b"] > weak["b"]
    assert strong["b"] <= 1.0

    assert expand_links([], adjacency) == {}
    assert expand_links(["a"], {}) == {}

    cyclic = {"a": [("b", "related_to")], "b": [("a", "related_to")]}
    assert isinstance(expand_links(["a"], cyclic, max_hops=5), dict)

    wide = {"seed": [(f"n{i}", "related_to") for i in range(100)]}
    assert len(expand_links(["seed"], wide, max_hops=1, budget=10)) <= 10

    assert expand_links(["a"], adjacency, max_hops=5, threshold=0.9) == {}


# ── Temporal parsing ──────────────────────────────────────────────────────

def test_temporal_extraction():
    now = datetime(2026, 6, 15, 12, 0, 0)

    assert extract_temporal_constraint("how does the parser work", now) is None
    assert extract_temporal_constraint("", now) is None
    assert extract_temporal_constraint(None, now) is None

    start, end = extract_temporal_constraint("what did I do yesterday", now)
    assert start.date() == datetime(2026, 6, 14).date() and end.date() == datetime(2026, 6, 14).date()
    assert start.hour == 0 and end.hour == 23

    start, end = extract_temporal_constraint("changes from last week", now)
    assert (end.date() - start.date()).days == 6
    assert end.date() < now.date()

    start, end = extract_temporal_constraint("the March 2026 migration", now)
    assert start.date() == datetime(2026, 3, 1).date() and end.date() == datetime(2026, 3, 31).date()

    start, end = extract_temporal_constraint("what shipped in 2024", now)
    assert start.date() == datetime(2024, 1, 1).date() and end.date() == datetime(2024, 12, 31).date()

    port_result = extract_temporal_constraint("error on port 9999", now)
    assert port_result is None or port_result[0].year != 9999

    start, end = extract_temporal_constraint("last 3 days of work", now)
    assert (now - start).days <= 4 and end >= now.replace(hour=0)

    result = extract_temporal_constraint("2 weeks ago", now)
    assert result is not None
    start, end = result
    target = now - timedelta(days=14)
    assert start <= target <= end, f"start={start} target={target} end={end}"
    assert (end - start).days >= 2

    start, end = extract_temporal_constraint("what happened today", now)
    assert start.date() == now.date() == end.date()

    start, end = extract_temporal_constraint("recently discussed", now)
    assert (end - start).days >= 13

    start, end = extract_temporal_constraint("the last month rollout", now)
    assert start.date() == datetime(2026, 5, 1).date() and end.date() == datetime(2026, 5, 31).date()

    start, end = extract_temporal_constraint("last year revenue", now)
    assert start.year == 2025 and end.year == 2025


def test_analyze_query():
    now = datetime(2026, 6, 15, 12, 0, 0)

    window, residual = analyze_query("gateway issues yesterday", now)
    assert window is not None
    assert "yesterday" not in residual.lower()
    assert "gateway" in residual and "issues" in residual

    window, residual = analyze_query("how does the parser work", now)
    assert window is None and residual == "how does the parser work"

    window, residual = analyze_query("what happened yesterday", now)
    assert residual.strip() != ""

    window, residual = analyze_query("the March 2026 migration", now)
    assert "2026" not in residual
    assert "migration" in residual

    window, residual = analyze_query("deploys in the last 3 days", now)
    assert "3 days" not in residual
    assert "deploys" in residual

    assert analyze_query("", now) == (None, "")
    assert analyze_query(None, now)[1] is None


def test_temporal_proximity():
    start, end = datetime(2026, 6, 1), datetime(2026, 6, 11)
    assert close(temporal_proximity(datetime(2026, 6, 6), start, end), 1.0)
    assert close(temporal_proximity(start, start, end), 0.0)
    assert close(temporal_proximity(datetime(2027, 1, 1), start, end), 0.0)
    assert close(temporal_proximity(None, start, end), 0.5)
    assert close(temporal_proximity(start, start, start), 1.0)
    mid_ish = temporal_proximity(datetime(2026, 6, 4), start, end)
    assert 0.0 < mid_ish < 1.0


def test_temporal_coverage():
    start, end = datetime(2026, 6, 1), datetime(2026, 6, 30)
    # Nine rows clustered in two days plus one far away; a plain score sort
    # would return only the cluster.
    rows = [{"_date": datetime(2026, 6, 2), "score": 0.9 - i * 0.01, "id": f"c{i}"} for i in range(9)]
    rows.append({"_date": datetime(2026, 6, 28), "score": 0.5, "id": "late"})

    selected = select_with_temporal_coverage(rows, 3, start, end)
    assert len(selected) == 3
    assert any(r["id"] == "late" for r in selected)

    assert len(select_with_temporal_coverage(rows[:2], 5, start, end)) == 2
    assert select_with_temporal_coverage(rows, 0, start, end) == []
    assert len(select_with_temporal_coverage(
        [{"score": 0.5, "id": str(i)} for i in range(5)], 2, start, end)) == 2


# ── Text utilities ────────────────────────────────────────────────────────

def test_trigram():
    assert close(trigram_similarity("hello world", "hello world"), 1.0)
    assert close(trigram_similarity("aaa", "zzz"), 0.0)
    assert close(trigram_similarity("Hello", "hello"), 1.0)
    assert 0.0 < trigram_similarity("hello world", "hello there") < 1.0
    assert close(trigram_similarity("", ""), 0.0)
    assert close(trigram_similarity("abc def", "def ghi"), trigram_similarity("def ghi", "abc def"))
    assert all(0.0 <= trigram_similarity(a, b) <= 1.0
               for a, b in (("a", "b"), ("test", "testing"), ("x y z", "z y x")))


def test_degenerate():
    for bad in (None, "", "   ", "...", "-", "n/a", "N/A", "null", "***", ".,;"):
        assert is_degenerate(bad), f"should reject {bad!r}"
    for good in ("a real memory", "x = 1", "OK", "42"):
        assert not is_degenerate(good), f"should accept {good!r}"


# ── Contradiction detection ───────────────────────────────────────────────

def test_contradiction():
    hit, conf = detect_contradiction(
        "The project does not use webpack for bundling assets",
        "The project uses webpack for bundling assets",
    )
    assert hit
    assert 0.5 <= conf <= 0.95

    hit, _ = detect_contradiction(
        "Caching is disabled for the session store",
        "Caching is enabled for the session store",
    )
    assert hit

    hit, _ = detect_contradiction(
        "The project uses webpack for bundling",
        "The project uses webpack for bundling",
    )
    assert not hit

    hit, _ = detect_contradiction("Completely unrelated topic here", "The sky is blue today")
    assert not hit

    hit, _ = detect_contradiction("", "something")
    assert not hit
    hit, _ = detect_contradiction(None, None)
    assert not hit

    hit, _ = detect_contradiction(
        "The build is fast and reliable now",
        "The build is fast and reliable now indeed",
    )
    assert not hit

    _, conf = detect_contradiction(
        "Feature flags are not enabled in production",
        "Feature flags are disabled in production",
    )
    assert conf <= 0.95
