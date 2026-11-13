"""Hindsight-derived retrieval and consolidation primitives for the Rekal engine.

This module is intentionally dependency-free (stdlib only) and side-effect free:
every function is pure so it can be unit-tested without a database, an LLM, or an
embedding backend.  `mempalace_rekal_engine` composes these primitives; nothing
here reaches back into the engine.

The algorithms are adapted from the Hindsight memory system.  Where Hindsight
targets Postgres + pgvector at scale, the adaptations below target the Rekal
SQLite store and its existing FTS5 + Chroma hybrid, keeping the same underlying
design principles:

  * Rank-space fusion (RRF) instead of score-space blending, because the arms'
    native scores (FTS5 bm25, cosine distance, graph activation, recency) are
    mutually incommensurable.
  * Rank-space strategy boosts, never score-space weights, because a weight
    large enough to matter also exceeds the dynamic range of 1/(k+rank) and
    collapses the ordering into "boosted arm first".
  * Multiplicative, bounded secondary signals so their influence stays
    proportional to base relevance regardless of the primary scorer's
    calibration.
  * Evidence accumulation (proof counts) rather than a free-floating confidence
    number that nothing can validate.

All scoring signals in this module use the convention that **0.5 is neutral**:
a signal of 0.5 leaves a score unchanged, above 0.5 boosts, below 0.5 penalizes.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import datetime, timedelta, timezone

# ══════════════════════════════════════════════════════════════════════════
#  Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════

# Hindsight uses k=60, the value from the original RRF paper. k damps the
# influence of rank differences deep in a list: the gap between rank 1 and 2 is
# large, between rank 200 and 201 negligible.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(result_lists, k=DEFAULT_RRF_K):
    """Fuse ranked candidate lists by rank position rather than by score.

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
#  Recency with a neutral midpoint
# ══════════════════════════════════════════════════════════════════════════

RECENCY_LINEAR_WINDOW_DAYS = 365.0
RECENCY_HALFLIFE_DAYS = 90.0


def compute_recency_decay(days_ago, function="linear", linear_window_days=None, halflife_days=None):
    """Age -> freshness signal in [0,1] where 0.5 is neutral.

    Unlike the engine's original `_recency_score` (pure exponential, 1.0 at age
    zero), this returns a *signal* meant for multiplicative combination, so the
    midpoint matters more than the endpoints.  Both are kept: the original drives
    the legacy weighted-sum path, this one drives the fusion path.
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
    """True when [start, end] covers exactly one *aligned* calendar month or year.

    Alignment is what separates a coarse date from a sliding window.  The
    original length-only test accepted any span of about a month, so a 30-day
    "past 30 days" window starting on the 13th classified as a coarse date and
    every result of that query had its recency capped at neutral.  A coarse
    date is a period a user *named* ("March 2026", "last year"), so it starts on
    the first day of the period and covers its last one.
    """
    if start is None or end is None:
        return False
    if start.hour or start.minute or start.second or start.microsecond:
        return False
    if start.day == 1:
        last_day = calendar.monthrange(start.year, start.month)[1]
        month_end = start.replace(day=last_day)
        if month_end <= end < month_end + timedelta(days=1):
            return True
    if (start.month, start.day) == (1, 1):
        year_end = datetime(start.year, 12, 31)
        if year_end <= end < year_end + timedelta(days=2):
            return True
    return False


def recency_for_range(occurred_start, occurred_end, fallback, now, function="linear",
                      linear_window_days=None, halflife_days=None):
    """Recency signal for a memory that may carry a coarse date range.

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


# ══════════════════════════════════════════════════════════════════════════
#  Graph / link expansion
# ══════════════════════════════════════════════════════════════════════════

# Saturating so the first shared connection carries most of the signal:
#   1 link -> 0.46   2 -> 0.76   3 -> 0.91   4 -> 0.96
ENTITY_SATURATION_SCALE = 0.5

# Per-hop decay for spreading activation, and the floor below which a node stops
# propagating. Both bound BFS cost as much as they shape relevance.
SPREAD_DECAY = 0.7
SPREAD_THRESHOLD = 0.2

# Relation multipliers: an explicit supersedes/contradicts edge is a much
# stronger statement about relatedness than a generic association.
RELATION_BOOST = {
    "supersedes": 2.0,
    "contradicts": 1.5,
    "related_to": 1.0,
}


def link_activation(shared_count, scale=ENTITY_SATURATION_SCALE):
    """Shared-connection count -> saturating activation in [0,1)."""
    return math.tanh(max(0, shared_count) * scale)


def expand_links(seed_ids, adjacency, max_hops=2, budget=50,
                 decay=SPREAD_DECAY, threshold=SPREAD_THRESHOLD):
    """Spreading activation over the memory-link graph.

    `adjacency` maps memory id -> iterable of (neighbour_id, relation).  Returns
    {memory_id: activation} for newly reached nodes only; seeds are excluded
    because they already entered retrieval through another arm and re-scoring
    them here would double-count.

    Activation propagates as `parent * relation_boost * decay`, and a node stops
    expanding once it drops below `threshold`.  Both `budget` and `max_hops`
    bound the traversal independently so a densely linked store cannot stall a
    search.
    """
    if not seed_ids or not adjacency:
        return {}

    visited = set(seed_ids)
    activations = {}
    frontier = [(sid, 1.0) for sid in seed_ids]

    for _ in range(max(1, max_hops)):
        if not frontier or len(activations) >= budget:
            break
        next_frontier = []
        for node_id, parent_activation in frontier:
            for neighbour_id, relation in adjacency.get(node_id, ()):
                if neighbour_id in visited:
                    continue
                boost = RELATION_BOOST.get(relation, 1.0)
                propagated = parent_activation * boost * decay
                if propagated <= threshold:
                    continue
                # A node reachable by several paths keeps its strongest.
                if propagated > activations.get(neighbour_id, 0.0):
                    activations[neighbour_id] = min(1.0, propagated)
                if len(activations) >= budget:
                    break
                next_frontier.append((neighbour_id, propagated))
            if len(activations) >= budget:
                break
        visited.update(node_id for node_id, _ in next_frontier)
        frontier = next_frontier

    return activations


# ══════════════════════════════════════════════════════════════════════════
#  Temporal query analysis
# ══════════════════════════════════════════════════════════════════════════

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Cheap pre-filter: locale-aware date parsing is expensive and is slowest
# precisely when there is no date to find. Deliberately an over-approximation.
_TEMPORAL_HINT = re.compile(
    r"\b(\d{4}|today|yesterday|tonight|now|recent|recently|last|past|previous|this|ago|"
    r"week|weeks|month|months|year|years|day|days|hour|hours|since|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)

_YEAR_ONLY = re.compile(r"\b(?:in|during|throughout|year)\s+(?P<year>\d{4})\b(?![-/.]\d)", re.IGNORECASE)
_MONTH_YEAR = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_N_UNITS_AGO = re.compile(r"\b(?P<n>\d+)\s+(?P<unit>hour|day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_LAST_N_UNITS = re.compile(r"\b(?:last|past|previous)\s+(?P<n>\d+)\s+(?P<unit>hour|day|week|month|year)s?\b", re.IGNORECASE)

_UNIT_DAYS = {"hour": 1.0 / 24.0, "day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}


def _day_range(start, end):
    """Expand a pair of dates to a full-day-aligned inclusive window."""
    return (
        start.replace(hour=0, minute=0, second=0, microsecond=0),
        end.replace(hour=23, minute=59, second=59, microsecond=999999),
    )


def extract_temporal_constraint(query, reference_date=None):
    """Parse a natural-language time window out of a query.

    Returns `(start, end)` as naive UTC datetimes, or None when the query has no
    temporal intent.  Rule-based on purpose: the patterns that actually appear in
    agent queries are few and closed, and a rule table is deterministic,
    debuggable, and costs microseconds where a parsing library costs
    milliseconds and a model call costs seconds.
    """
    if not query:
        return None
    if not _TEMPORAL_HINT.search(query):
        return None

    now = reference_date or datetime.now(timezone.utc).replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    low = query.lower()

    # Explicit "<month> <year>" beats a bare year in the same query.
    match = _MONTH_YEAR.search(low)
    if match:
        month = _MONTHS[match.group("month")]
        year = int(match.group("year"))
        if 1 <= month <= 12 and _plausible_year(year, now):
            last_day = calendar.monthrange(year, month)[1]
            return _day_range(datetime(year, month, 1), datetime(year, month, last_day))

    match = _YEAR_ONLY.search(low)
    if match:
        year = int(match.group("year"))
        if _plausible_year(year, now):
            return _day_range(datetime(year, 1, 1), datetime(year, 12, 31))

    match = _N_UNITS_AGO.search(low)
    if match:
        days = int(match.group("n")) * _UNIT_DAYS[match.group("unit")]
        target = now - timedelta(days=days)
        # "3 days ago" means around then, not exactly then.
        pad = max(0.5, days * 0.15)
        return _day_range(target - timedelta(days=pad), target + timedelta(days=pad))

    match = _LAST_N_UNITS.search(low)
    if match:
        days = int(match.group("n")) * _UNIT_DAYS[match.group("unit")]
        return _day_range(now - timedelta(days=days), now)

    if "day before yesterday" in low:
        day = now - timedelta(days=2)
        return _day_range(day, day)
    if "yesterday" in low:
        day = now - timedelta(days=1)
        return _day_range(day, day)
    if "today" in low or "tonight" in low or "this morning" in low or "this afternoon" in low:
        return _day_range(now, now)
    if "last week" in low:
        start = now - timedelta(days=now.weekday() + 7)
        return _day_range(start, start + timedelta(days=6))
    if "this week" in low:
        start = now - timedelta(days=now.weekday())
        return _day_range(start, now)
    if "last month" in low:
        first_this = now.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        return _day_range(last_prev.replace(day=1), last_prev)
    if "this month" in low:
        return _day_range(now.replace(day=1), now)
    if "last year" in low:
        return _day_range(datetime(now.year - 1, 1, 1), datetime(now.year - 1, 12, 31))
    if "this year" in low:
        return _day_range(datetime(now.year, 1, 1), now)
    if "a few days ago" in low:
        return _day_range(now - timedelta(days=5), now - timedelta(days=2))
    if "a couple of days ago" in low or "a couple days ago" in low:
        return _day_range(now - timedelta(days=3), now - timedelta(days=1))
    if "recently" in low or "recent" in low:
        return _day_range(now - timedelta(days=14), now)

    return None


def _plausible_year(year, reference):
    """Reject port numbers and version strings masquerading as years."""
    return max(1, reference.year - 120) <= year <= reference.year + 20


# Phrases that express *when* rather than *what*. Removing them from the
# lexical query matters because FTS5 combines terms with AND: leaving
# "yesterday" in "gateway issues yesterday" requires a stored memory to contain
# the literal word "yesterday", which is almost never true, so the entire
# keyword arm silently returns nothing.
_TEMPORAL_PHRASES = re.compile(
    r"\b("
    r"day before yesterday|a couple of days ago|a couple days ago|a few days ago|"
    r"(?:last|past|previous|next|this)\s+\d+\s+(?:hour|day|week|month|year)s?|"
    r"\d+\s+(?:hour|day|week|month|year)s?\s+ago|"
    r"(?:last|past|previous|this)\s+(?:week|month|year|night)|"
    r"(?:in|during|throughout|year)\s+\d{4}|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}|"
    r"yesterday|today|tonight|recently|recent|this morning|this afternoon"
    r")\b",
    re.IGNORECASE,
)


def analyze_query(query, reference_date=None):
    """Split a query into a time window and the residual lexical query.

    Returns `(window, residual)` where `window` is `(start, end)` or None and
    `residual` is the query with temporal phrases removed.  When no temporal
    intent is found the original query is returned unchanged, so callers can
    use the residual unconditionally.
    """
    window = extract_temporal_constraint(query, reference_date)
    if not query:
        return window, query
    if window is None:
        return None, query
    residual = _TEMPORAL_PHRASES.sub(" ", query)
    residual = re.sub(r"\s+", " ", residual).strip()
    # Never hand back an empty lexical query: a purely temporal question like
    # "what happened yesterday" still needs the original text for the vector
    # arm, which has no AND semantics to be defeated by.
    return window, residual or query


def temporal_proximity(target, window_start, window_end):
    """Triangular kernel: 1.0 at the window midpoint, 0.0 at its edges.

    Rewards the centre rather than the edges because a window derived from
    "last week" is an estimate; something dated exactly at the boundary is as
    likely to be outside the user's intent as inside it.
    """
    if target is None or window_start is None or window_end is None:
        return 0.5
    total = (window_end - window_start).total_seconds()
    if total <= 0:
        return 1.0
    midpoint = window_start + (window_end - window_start) / 2
    distance = abs((target - midpoint).total_seconds())
    return max(0.0, 1.0 - min(distance / (total / 2), 1.0))


def select_with_temporal_coverage(rows, limit, window_start, window_end,
                                  date_key="_date", score_key="score", buckets=8):
    """Pick `limit` rows spread across the time window instead of clustered.

    Rows are bucketed by position in the window, then drained round-robin: every
    populated bucket contributes its best row before any contributes a second.
    Without this, a similarity-ordered slice of "what happened last month"
    collapses onto whichever few days dominate the store.
    """
    if limit <= 0:
        return []
    if len(rows) <= limit:
        return list(rows)

    ranked = sorted(rows, key=lambda r: r.get(score_key, 0.0), reverse=True)
    total = (window_end - window_start).total_seconds() if window_start and window_end else 0

    grouped = {}
    for row in ranked:
        index = 0
        date = row.get(date_key)
        if date is not None and total > 0:
            fraction = (date - window_start).total_seconds() / total
            index = max(0, min(int(fraction * buckets), buckets - 1))
        grouped.setdefault(index, []).append(row)

    selected = []
    tier = 0
    while len(selected) < limit and any(len(group) > tier for group in grouped.values()):
        tier_rows = [group[tier] for group in grouped.values() if len(group) > tier]
        tier_rows.sort(key=lambda r: r.get(score_key, 0.0), reverse=True)
        for row in tier_rows:
            if len(selected) >= limit:
                break
            selected.append(row)
        tier += 1
    return selected


# ══════════════════════════════════════════════════════════════════════════
#  Text similarity and quality
# ══════════════════════════════════════════════════════════════════════════

_TRGM_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def trigram_set(text):
    """Padded per-word trigram set, matching PostgreSQL pg_trgm semantics."""
    trigrams = set()
    for word in _TRGM_WORD.findall((text or "").lower()):
        padded = f"  {word} "
        for i in range(len(padded) - 2):
            trigrams.add(padded[i : i + 3])
    return trigrams


def trigram_similarity(a, b):
    """Jaccard similarity over trigram sets, in [0,1]."""
    set_a = a if isinstance(a, set) else trigram_set(a)
    set_b = b if isinstance(b, set) else trigram_set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union if union else 0.0


_DEGENERATE = {"...", "…", "-", "--", "---", ".", "..", "•", "·", "*", "**", "***", "n/a", "na", "none", "null"}
_PUNCT_ONLY = set(".,;:!?-–—…\"'`´ \t\n\r")


def is_degenerate(text):
    """True for content that carries no recoverable meaning.

    Filters extraction artefacts before they occupy a memory slot and, worse,
    match every future query weakly through FTS prefix expansion.
    """
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.lower() in _DEGENERATE:
        return True
    if all(ch in _PUNCT_ONLY for ch in stripped):
        return True
    if len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
#  Contradiction detection
# ══════════════════════════════════════════════════════════════════════════

# Surface markers of negation/reversal. Lexical detection cannot prove a
# contradiction, so callers treat a hit as a candidate for review (a
# `contradicts` link) rather than as grounds to delete or rewrite anything.
NEGATION_MARKERS = (
    "not ", "no longer ", "doesn't ", "does not ", "isn't ", "is not ",
    "aren't ", "are not ", "won't ", "will not ", "never ", "cannot ", "can't ",
    "removed ", "stopped ", "dropped ", "disabled ", "deprecated ", "reverted ",
    "instead of ", "rather than ", "switched from ", "moved away from ",
)

# Antonym pairs that flip a statement's meaning without any negation word.
POLARITY_PAIRS = (
    ("enable", "disable"), ("enabled", "disabled"), ("allow", "deny"),
    ("allowed", "denied"), ("add", "remove"), ("added", "removed"),
    ("include", "exclude"), ("start", "stop"), ("started", "stopped"),
    ("on", "off"), ("true", "false"), ("always", "never"),
    ("support", "unsupported"), ("prefer", "avoid"),
)

CONTRADICTION_MIN_SIMILARITY = 0.35
CONTRADICTION_MAX_SIMILARITY = 0.97


def _has_negation(text):
    padded = f" {text.lower()} "
    return any(marker in padded for marker in NEGATION_MARKERS)


def _polarity_conflict(a_low, b_low):
    a_words = set(_TRGM_WORD.findall(a_low))
    b_words = set(_TRGM_WORD.findall(b_low))
    for left, right in POLARITY_PAIRS:
        if (left in a_words and right in b_words) or (right in a_words and left in b_words):
            return True
    return False


# ── Value substitution (numbers, named entities) ──────────────────────────
#
# Two statements can disagree without any negation or antonym: the subject and
# predicate are identical and only the *value* differs.  "TTL is 300s" against
# "TTL is 600s" is lexically nearly identical and identically polarised, so both
# detectors above miss it -- and it is exactly the kind of change the operator
# most wants flagged.
#
# These detectors are deliberately high precision.  A miss costs one review
# opportunity; a false conflict fills the review queue with noise and, worse,
# teaches the operator to stop reading it.  Each detector therefore requires the
# two statements to collapse to the *same skeleton* once the differing slot is
# masked: same number of slots, decorated by identical surrounding words, with
# exactly one differing value.  Anything looser is left to human review.

# Only used for the optional fuzzy fallback on numeric skeletons; kept far above
# the ordinary similarity band so it cannot turn "related sentences with
# numbers" into conflicts.
VALUE_CONFLICT_SKELETON_SIMILARITY = 0.97

_NUMBER_TOKEN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])")
# A whole capitalised phrase ("Session TTL"), so a substitution can be told
# apart from a differently-scoped subject: see _proper_name_conflict.
_PROPER_RUN = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:\s+[A-Z][A-Za-z0-9_]*)*\b")
_SKELETON_DROP = {"the", "a", "an"}

# Numbers that name a *slot* rather than state a value: "filler note number 3",
# "step 4", "build 91", "item 12".  An index is not a fact, so two memories
# that differ only in their index are two members of a series, not a
# disagreement -- and stores are full of them.  Without this guard, bulk
# ingestion of numbered notes links every pair it writes as contradictory,
# which is the single worst false positive this detector can produce because it
# fires O(n^2) over a batch.
_ENUMERATOR_PREFIXES = frozenset({
    "number", "no", "no.", "#", "item", "index", "idx", "step", "part",
    "entry", "row", "line", "page", "chapter", "section", "version", "ver",
    "build", "rev", "revision", "id", "note", "notes", "attempt", "try",
    "level", "phase", "task", "issue", "ticket", "case", "run", "batch",
    "round", "slot", "position", "positional",
})


def _skeleton_tokens(text):
    """Words of `text` with articles dropped, punctuation collapsed.

    Splitting on any non-alphanumeric character is what makes "300s" and
    "300 s" produce the same skeleton, and what keeps the mask placeholder a
    token of its own in both.
    """
    words = re.split(r"[^a-z0-9#]+", text.lower())
    return [w for w in words if w and w not in _SKELETON_DROP]


def _looks_ordinal(text):
    """True when a number in `text` is an index rather than a stated value."""
    for match in _NUMBER_TOKEN.finditer(text):
        prefix = text[: match.start()].rstrip(" .:#")
        words = re.split(r"[^a-z0-9#]+", prefix.lower())
        previous = words[-1] if words else ""
        if previous in _ENUMERATOR_PREFIXES:
            return True
        following = text[match.end():].lstrip()
        if following[:3].lower() == "of ":
            return True
    return False


def _numeric_conflict(a, b):
    """True when both sides carry numbers and only the numbers differ."""
    values_a = _NUMBER_TOKEN.findall(a)
    values_b = _NUMBER_TOKEN.findall(b)
    if not values_a or not values_b:
        return False
    if values_a == values_b:
        return False
    if len(values_a) != len(values_b):
        return False
    if _looks_ordinal(a) or _looks_ordinal(b):
        return False

    skeleton_a = _skeleton_tokens(_NUMBER_TOKEN.sub(" # ", a))
    skeleton_b = _skeleton_tokens(_NUMBER_TOKEN.sub(" # ", b))
    if "#" not in skeleton_a or "#" not in skeleton_b:
        return False
    # At least one real word of context: a bare number is not a statement.
    if not any(w != "#" and len(w) >= 2 for w in skeleton_a):
        return False
    if skeleton_a == skeleton_b:
        return True
    if len(skeleton_a) != len(skeleton_b):
        return False
    return trigram_similarity(" ".join(skeleton_a), " ".join(skeleton_b)) >= VALUE_CONFLICT_SKELETON_SIMILARITY


def _proper_name_conflict(a, b):
    """True when exactly one single-token capitalised name is substituted.

    The position and the surrounding wording must both match exactly, and the
    substituted slot must be one word on each side.  That second condition is
    what keeps the detector from firing on a *different subject*: "Session TTL
    is 300s" and "Cache TTL is 300s" describe two separate settings, while
    "Alice owns billing" and "Bob owns billing" cannot both hold for one
    billing system.  Lexically the two cases differ only by whether the changed
    phrase is a single name or a name-plus-noun, so that is the discriminator.
    """
    phrases_a = _PROPER_RUN.findall(a)
    phrases_b = _PROPER_RUN.findall(b)
    if not phrases_a or len(phrases_a) != len(phrases_b):
        return False

    diffs = [(left, right) for left, right in zip(phrases_a, phrases_b) if left != right]
    if len(diffs) != 1:
        return False
    left, right = diffs[0]
    if " " in left or " " in right:
        return False

    def _mask(text):
        out = []
        last = 0
        for match in _PROPER_RUN.finditer(text):
            out.append(text[last:match.start()])
            out.append(" # ")
            last = match.end()
        out.append(text[last:])
        return _skeleton_tokens("".join(out))

    mask_a, mask_b = _mask(a), _mask(b)
    if mask_a != mask_b:
        return False
    # "# owns billing" is a shared predicate; a lone "#" is not.
    return len(mask_a) >= 3


def _value_conflict(new_content, existing_content):
    """Detect a value/number/entity substitution between similar statements."""
    return (_numeric_conflict(new_content, existing_content)
            or _proper_name_conflict(new_content, existing_content))


def detect_contradiction(new_content, existing_content,
                         min_similarity=CONTRADICTION_MIN_SIMILARITY,
                         max_similarity=CONTRADICTION_MAX_SIMILARITY):
    """Decide whether two statements about the same subject disagree.

    Returns `(is_contradiction, confidence)`.

    The gate is deliberately two-sided.  Texts must be similar enough to be
    about the same thing, but not so similar that they are effectively the same
    sentence -- near-identical text is a duplicate, which is handled by content
    hashing, not a contradiction.  Within that band, either an explicit negation
    marker on exactly one side, an antonym pair across the two, or a single
    substituted value (a number or a capitalised name) is treated as a
    candidate conflict.

    Value substitution is checked last and only when neither negation nor a
    polarity flip fired, so the existing observations keep exactly the
    behaviour they had: the new detector can only add conflicts, never
    reinterpret a negation hit.
    """
    if not new_content or not existing_content:
        return False, 0.0

    similarity = trigram_similarity(new_content, existing_content)
    if similarity < min_similarity or similarity > max_similarity:
        return False, 0.0

    new_low = new_content.lower()
    existing_low = existing_content.lower()

    negation_differs = _has_negation(new_low) != _has_negation(existing_low)
    polarity_differs = _polarity_conflict(new_low, existing_low)

    if not negation_differs and not polarity_differs:
        if not _value_conflict(new_content, existing_content):
            return False, 0.0
        # Both sides are identically polarised and share their wording, so the
        # only reading is a changed value.  Confidence stays below the negation
        # band because lexical detection cannot tell an update from a
        # restatement of a different scope.
        return True, round(min(0.9, 0.62 + 0.3 * similarity), 3)

    # Confidence rises with topical overlap and with agreement between the two
    # independent detectors.
    confidence = 0.5 + 0.3 * similarity
    if negation_differs and polarity_differs:
        confidence += 0.15
    return True, round(min(0.95, confidence), 3)
