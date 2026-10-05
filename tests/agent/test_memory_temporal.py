"""Unit tests for the ported Hindsight temporal primitives.

Pure-function tests: no database, no network, no embedding backend.  Ported
from the upstream ``tests/plugins/memory/test_rekal_hindsight.py`` checks.
"""

from datetime import datetime, timedelta

from agent.memory.temporal import (
    analyze_query,
    extract_temporal_constraint,
    select_with_temporal_coverage,
    temporal_proximity,
)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


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
