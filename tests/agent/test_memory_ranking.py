"""Unit tests for the ported Hindsight ranking primitives.

Pure-function tests: no database, no network, no embedding backend.  Ported
from the upstream ``tests/plugins/memory/test_rekal_hindsight.py`` checks.
"""

import math
from datetime import datetime

from agent.memory.ranking import (
    BOOST_LEVELS,
    DEFAULT_RRF_K,
    boosted_rrf_score,
    combined_score,
    compute_recency_decay,
    proof_norm,
    reciprocal_rank_fusion,
    recency_for_range,
    spans_calendar_period,
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
