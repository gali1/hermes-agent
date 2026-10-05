"""Temporal query analysis: window extraction and coverage-aware selection.

Ported verbatim from the OpenCode/MemPalace Hindsight primitives (upstream
``plugins/memory/rekal/mempalace_hindsight.py``).  Stdlib only and pure.
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone

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

    Ported from the OpenCode/MemPalace Hindsight primitives.

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

    Ported from the OpenCode/MemPalace Hindsight primitives.

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

    Ported from the OpenCode/MemPalace Hindsight primitives.

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

    Ported from the OpenCode/MemPalace Hindsight primitives.

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
