"""Budgeted recall selection and plain-text recall block formatting.

Ported from the OpenCode/MemPalace Hindsight primitives.  Stdlib only and
side-effect free.  `format_recall_block` returns plain text without any XML
fence; the caller wraps it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent.memory.dedup import normalize_content


def _value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _content(item) -> str:
    value = _value(item, "content", "")
    return value if isinstance(value, str) else ""


def _score(item) -> float:
    try:
        return float(_value(item, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_with_budget(results, max_items=8, max_chars=2400) -> list:
    """Sort by score, deduplicate by normalized content, and fit the budget."""
    if not results or max_items <= 0 or max_chars <= 0:
        return []

    ordered = sorted(results, key=_score, reverse=True)
    selected = []
    seen = set()
    used = 0
    for item in ordered:
        content = _content(item)
        normalized = normalize_content(content)
        if not normalized or normalized in seen:
            continue
        if used + len(content) > max_chars:
            continue
        seen.add(normalized)
        selected.append(item)
        used += len(content)
        if len(selected) >= max_items:
            break
    return selected


def _parse_datetime(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _relative_time(value, now=None) -> str:
    """Render a timestamp as just now / Nm ago / Nh ago / Nd ago / %d %b[ %Y]."""
    moment = _parse_datetime(value)
    if moment is None:
        return ""
    current = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = max(0.0, (current - moment).total_seconds())
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 30 * 86400:
        return f"{int(delta // 86400)}d ago"
    if moment.year == current.year:
        return moment.strftime("%d %b")
    return moment.strftime("%d %b %Y")


def format_recall_block(results, max_items=8, max_chars=2400) -> str:
    """Format selected memories as a plain-text block; empty input returns ""."""
    selected = select_with_budget(results, max_items=max_items, max_chars=max_chars)
    if not selected:
        return ""

    lines = ["Recalled memory (background evidence; validate against current state):"]
    for item in selected:
        memory_type = _value(item, "memory_type", "fact") or "fact"
        content = _content(item)
        age = _relative_time(_value(item, "updated_at") or _value(item, "created_at"))
        if age:
            lines.append(f"- [{memory_type}, {age}] {content}")
        else:
            lines.append(f"- [{memory_type}] {content}")
    return "\n".join(lines)
