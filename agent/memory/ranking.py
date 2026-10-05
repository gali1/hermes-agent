"""Rank-space fusion, recency decay, and bounded combined scoring.

Ported verbatim from the OpenCode/MemPalace Hindsight primitives (upstream
``plugins/memory/rekal/mempalace_hindsight.py``).  Stdlib only and pure:
every function is deterministic and side-effect free.

All scoring signals in this module use the convention that **0.5 is neutral**:
a signal of 0.5 leaves a score unchanged, above 0.5 boosts, below 0.5 penalizes.
"""

from __future__ import annotations

import calendar
import math
import re

# RRF damping constant.  Hindsight inherited k=60 from the original RRF paper,
# where result lists are hundreds deep and only the head matters.  A personal
# memory store fuses a handful of short arms (fts/vec/graph/temporal), so k=60
# flattens rank-1 vs rank-2 to 1/61 vs 1/62 and lets secondary arms tie the
# lexical winner.  k=20 keeps head differences meaningful while still damping
# the long tail (rank 100 contributes ~0.008 vs ~0.006 at k=60).
DEFAULT_RRF_K = 20

# When a query classifies as a lexical lookup, the fused list is anchored on
# its best BM25 candidate when that candidate is a genuinely strong match:
# either at least LEXICAL_ANCHOR_MIN_FTS normalized strength, or leading the
# runner-up by LEXICAL_ANCHOR_MARGIN.  1/(1+exp(bm25)); 0.6 corresponds to
# bm25 <= -0.4.  Question-like queries are never anchored.
LEXICAL_ANCHOR_MIN_FTS = 0.6
LEXICAL_ANCHOR_MARGIN = 0.15

# Hard ceiling for the vector arm's contribution to the additive relevance
# score (enforced by ``MemoryStore._resolve_weights``).  Embeddings are
# high-recall but low-precision: above this share, semantically-adjacent noise
# can outrank exact lexical matches.  The separate ``vector_max_share`` cap in
# the store bounds how many vector-only memories may appear near the top of a
# fused result list.
MAX_VECTOR_WEIGHT = 0.4


def reciprocal_rank_fusion(result_lists, k=DEFAULT_RRF_K):
    """Fuse ranked candidate lists by rank position rather than by score.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    `result_lists` is an ordered sequence of `(arm_name, [ids...])` pairs, each
    inner list ordered best-first.  Returns a list of dicts ordered by fused
    score descending:

        {"id", "rrf_score", "rrf_rank", "source_ranks": {f"{arm}_rank": rank}}

    score(d) = sum over arms of 1 / (k + rank_in_arm(d)), rank being 1-based.

    A document found by several arms accumulates several reciprocal terms, so
    convergent evidence outranks a single arm's favourite without ever needing
    the arms' raw scores to be comparable.

    Ties preserve first-appearance order (arms are walked in the order given,
    ranks ascending), matching Python's stable sort over insertion-ordered dicts.
    """
    rrf_scores = {}
    source_ranks = {}
    order = []

    for arm_name, ids in result_lists:
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                source_ranks[doc_id] = {}
                order.append(doc_id)
            rrf_scores[doc_id] += 1.0 / (k + rank)
            source_ranks[doc_id][f"{arm_name}_rank"] = rank

    ranked = sorted(order, key=lambda d: rrf_scores[d], reverse=True)
    return [
        {
            "id": doc_id,
            "rrf_score": rrf_scores[doc_id],
            "rrf_rank": position,
            "source_ranks": source_ranks[doc_id],
        }
        for position, doc_id in enumerate(ranked, start=1)
    ]


# Rank divisors, not score multipliers.  Boosting rank r to r/divisor means a
# boosted candidate at rank r outranks an unboosted one at rank s whenever
# r < divisor * s -- independent of k and of the candidate-pool size.  A
# score-space weight w would instead have reach w*(k+s)-k, which is dominated by
# w*k at the head of the list and degenerates into a lexicographic sort.
BOOST_LEVELS = {
    "low": 2.0,
    "medium": 4.0,
    "high": 8.0,
}


def boosted_rrf_score(rrf_score, source_ranks, boosts, k=DEFAULT_RRF_K):
    """Apply rank-space per-arm boosts to a fused score.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    `boosts` maps arm name to a key of BOOST_LEVELS.  Unknown arms and unknown
    levels are ignored rather than raising, so a malformed configuration
    degrades to plain RRF instead of breaking retrieval.
    """
    if not boosts:
        return rrf_score
    delta = 0.0
    for arm, level in boosts.items():
        rank = source_ranks.get(f"{arm}_rank")
        if rank is None:
            continue
        divisor = BOOST_LEVELS.get(level)
        if not divisor:
            continue
        delta += 1.0 / (k + rank / divisor) - 1.0 / (k + rank)
    return rrf_score + delta


# ══════════════════════════════════════════════════════════════════════════
#  Query-aware strategy boosts
# ══════════════════════════════════════════════════════════════════════════

# Signals that a query is a lexical lookup (identifiers, code symbols, quoted
# strings, versions, hashes, filenames) rather than a natural-language question.
# Lexical lookups are answered by the FTS arm; conceptual questions are served
# by the vector and graph arms.  Pure RRF treats every arm equally and therefore
# trades away top-1 precision on lookups.
_IDENTIFIER_RE = re.compile(
    r"""
    `[^`]+`                                             # backticked span
    |"[^"]+"                                            # double-quoted phrase
    |'[^']+'                                            # single-quoted phrase
    |[A-Za-z_][A-Za-z0-9_]*[._/\\:][A-Za-z0-9_./\\:]+  # dotted/slashed/namespaced
    |[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+                 # snake_case
    |[a-z]+[A-Z][A-Za-z0-9]+                            # camelCase/PascalCase
    |\b[A-Z][A-Z0-9_]{2,}\b                             # ALL_CAPS constant
    |\bv?\d+\.\d+(?:\.\d+)?\b                           # version number
    |\b[0-9a-fA-F]{7,}\b                                # commit hash / hex id
    |\b[A-Za-z]+\.(?:py|js|ts|tsx|json|ya?ml|toml|md|sql|sh|go|rs|java|c|cpp|h)\b
    """,
    re.VERBOSE,
)

_QUESTION_WORDS = frozenset({
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "explain", "describe", "summarize", "summary", "history", "context",
    "reason", "rationale", "decide", "decided", "decision", "decisions",
    "preference", "preferences", "prefer", "approach", "remember",
    "discuss", "discussed", "know", "learned", "tell",
})


def infer_strategy_boosts(query):
    """Classify a query and return rank-space boosts for the RRF arms.

    Returns a dict suitable for :func:`boosted_rrf_score`, or an empty dict
    when no classification applies (plain RRF).  The classification is
    deliberately rule-based and cheap: it runs on every advanced retrieval.

    * natural-language questions -> vector/graph lead, FTS damped;
    * identifiers, quoted spans, filenames, versions, or <= 3 tokens ->
      FTS leads, vector damped;
    * longer statement-like phrases -> FTS leans up, vector damped;
    * anything else -> balanced.
    """
    q = (query or "").strip()
    if not q:
        return {}
    tokens = q.split()
    if not tokens:
        return {}

    first_word = tokens[0].strip("?,.:;!()[]{}\"'`").lower() if tokens else ""
    # Question words only count at the head of the query: a literal memory
    # sentence ("We decided to use PostgreSQL 17 ...") must not be classified
    # as a question just because it contains "decided" mid-sentence.
    question_like = q.rstrip().endswith("?") or first_word in _QUESTION_WORDS
    has_identifier = bool(_IDENTIFIER_RE.search(q))

    if question_like:
        return {"vec": "high", "graph": "medium", "fts": "low"}
    if has_identifier or len(tokens) <= 3:
        return {"fts": "high", "vec": "low"}
    if len(tokens) >= 6:
        return {"fts": "medium", "vec": "low"}
    return {"fts": "medium", "vec": "medium"}


# ══════════════════════════════════════════════════════════════════════════
#  Recency with a neutral midpoint
# ══════════════════════════════════════════════════════════════════════════

RECENCY_LINEAR_WINDOW_DAYS = 365.0
RECENCY_HALFLIFE_DAYS = 90.0

# A date range spanning exactly one calendar month or year is a coarse date
# ("sometime in 2015"), not a real interval. Tolerance absorbs the sub-second
# offsets used to keep same-batch memories individually ordered.
_CALENDAR_PERIOD_TOLERANCE_SECONDS = 86400.0


def compute_recency_decay(days_ago, function="linear", linear_window_days=None, halflife_days=None):
    """Age -> freshness signal in [0,1] where 0.5 is neutral.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    Unlike a pure exponential score (1.0 at age zero), this returns a *signal*
    meant for multiplicative combination, so the midpoint matters more than the
    endpoints.
    """
    window = linear_window_days if linear_window_days and linear_window_days > 0 else RECENCY_LINEAR_WINDOW_DAYS
    halflife = halflife_days if halflife_days and halflife_days > 0 else RECENCY_HALFLIFE_DAYS

    if function == "none":
        return 0.5
    if function == "exponential":
        # Clamp before the power: 0.5 ** (large negative) overflows.
        if days_ago <= 0:
            return 1.0
        return 0.5 ** (days_ago / halflife)
    return max(0.1, min(1.0, 1.0 - (days_ago / window)))


def spans_calendar_period(start, end):
    """True when [start, end] covers exactly one calendar month or year."""
    if start is None or end is None:
        return False
    span = (end - start).total_seconds()
    if span <= 0:
        return False
    month_seconds = calendar.monthrange(start.year, start.month)[1] * 86400.0
    year_seconds = (366.0 if calendar.isleap(start.year) else 365.0) * 86400.0
    return any(
        period - _CALENDAR_PERIOD_TOLERANCE_SECONDS <= span <= period
        for period in (month_seconds, year_seconds)
    )


def recency_for_range(occurred_start, occurred_end, fallback, now, function="linear",
                      linear_window_days=None, halflife_days=None):
    """Recency signal for a memory that may carry a coarse date range.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    Coarse dates are scored from the END of their period and capped at neutral.
    Scoring "the 2026 summit" (stored as 2026-01-01) from its start would read as
    eight months stale in August 2026; scoring it from the end without the cap
    would instead hand an in-progress period a freshness bonus it has not earned.
    """
    if occurred_start is not None and occurred_end is not None and spans_calendar_period(occurred_start, occurred_end):
        days = (now - occurred_end).total_seconds() / 86400.0
        return min(0.5, compute_recency_decay(days, function, linear_window_days, halflife_days))

    effective = occurred_start or fallback or occurred_end
    if effective is None:
        return 0.5
    days = (now - effective).total_seconds() / 86400.0
    return compute_recency_decay(days, function, linear_window_days, halflife_days)


# ══════════════════════════════════════════════════════════════════════════
#  Evidence strength
# ══════════════════════════════════════════════════════════════════════════


def proof_norm(proof_count):
    """Evidence strength -> signal in [0,1], neutral 0.5.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    Logarithmic so the tenth corroboration matters far less than the second.
    A single unreinforced memory sits exactly at neutral and is neither rewarded
    nor punished:

        1  -> 0.500      5  -> 0.661
        2  -> 0.569     10 -> 0.730
        3  -> 0.610     50 -> 0.891
    """
    try:
        count = int(proof_count or 1)
    except (TypeError, ValueError):
        return 0.5
    if count < 1:
        return 0.5
    return min(1.0, max(0.0, 0.5 + (math.log(count) / 10.0)))


# ══════════════════════════════════════════════════════════════════════════
#  Multiplicative combined scoring
# ══════════════════════════════════════════════════════════════════════════

# Each alpha caps its signal's influence at +-alpha/2.  Kept deliberately small:
# these are tie-breakers among candidates the primary scorer already considers
# relevant, not relevance signals in their own right.
#
# The four alphas are chosen together so the *combined* swing stays bounded:
#   max factor = 1.075 * 1.06 * 1.035 * 1.06 ~= 1.2501
#   min factor = 0.925 * 0.94 * 0.965 * 0.94 ~= 0.7887
#   ratio                                    ~= 1.585
# so any candidate whose base score leads by more than ~1.6x cannot be
# overtaken by secondary signals alone.  Raising these is the single easiest
# way to turn relevance ranking into a recency sort, so the bound is asserted
# by `test_combined_score` rather than left as a free-floating constant.
#
# Proof carries the smallest alpha because corroboration count is the least
# direct evidence of relevance to *this* query: a heavily reinforced memory is
# well-established, which is not the same as being what was asked about.
RECENCY_ALPHA = 0.15
IMPORTANCE_ALPHA = 0.12
PROOF_ALPHA = 0.07
GRAPH_ALPHA = 0.12


def combined_score(base, recency=0.5, importance=0.5, proof=0.5, graph=0.5,
                   recency_alpha=RECENCY_ALPHA, importance_alpha=IMPORTANCE_ALPHA,
                   proof_alpha=PROOF_ALPHA, graph_alpha=GRAPH_ALPHA):
    """Scale `base` by bounded multiplicative signals, each neutral at 0.5.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    Multiplicative rather than additive so a secondary signal's absolute effect
    stays proportional to base relevance: a weak candidate cannot be promoted
    past a strong one by recency alone, which is exactly what an additive term
    does when the base scores are tightly clustered.
    """
    factor = 1.0
    factor *= 1.0 + recency_alpha * (recency - 0.5)
    factor *= 1.0 + importance_alpha * (importance - 0.5)
    factor *= 1.0 + proof_alpha * (proof - 0.5)
    factor *= 1.0 + graph_alpha * (graph - 0.5)
    return base * factor
