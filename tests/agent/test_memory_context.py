"""Unit tests for budgeted recall selection and formatting."""

from datetime import datetime, timedelta, timezone

from agent.memory.context import format_recall_block, select_with_budget
from agent.memory.schema import MemoryResult


def test_select_with_budget_orders_and_dedups():
    results = [
        {"content": "low", "score": 0.1},
        {"content": "high", "score": 0.9},
        {"content": "mid", "score": 0.5},
    ]
    picked = select_with_budget(results, max_items=2, max_chars=100)
    assert [r["content"] for r in picked] == ["high", "mid"]

    dupes = [
        {"content": "Same  content", "score": 0.9},
        {"content": "same content", "score": 0.8},
    ]
    assert len(select_with_budget(dupes)) == 1

    assert select_with_budget([]) == []
    assert select_with_budget(None) == []
    assert select_with_budget(results, max_items=0) == []
    assert select_with_budget(results, max_items=3, max_chars=0) == []


def test_select_with_budget_char_budget():
    results = [
        {"content": "x" * 50, "score": 0.9},
        {"content": "small", "score": 0.5},
    ]
    picked = select_with_budget(results, max_items=5, max_chars=20)
    assert [r["content"] for r in picked] == ["small"]

    two = [
        {"content": "1234567890", "score": 0.9},
        {"content": "abcdefghij", "score": 0.8},
        {"content": "last", "score": 0.7},
    ]
    picked = select_with_budget(two, max_items=5, max_chars=20)
    assert [r["content"] for r in picked] == ["1234567890", "abcdefghij"]


def test_format_recall_block():
    now = datetime.now(timezone.utc)
    results = [
        {
            "content": "The parser uses recursive descent",
            "memory_type": "architecture",
            "score": 0.9,
            "created_at": (now - timedelta(minutes=5)).isoformat(),
        },
        {
            "content": "A very old note about the build",
            "memory_type": "fact",
            "score": 0.5,
            "created_at": (now - timedelta(days=400)).isoformat(),
        },
    ]
    block = format_recall_block(results)
    lines = block.splitlines()
    assert lines[0] == "Recalled memory (background evidence; validate against current state):"
    assert lines[1].startswith("- [architecture, ")
    assert "5m ago" in lines[1]
    assert "The parser uses recursive descent" in lines[1]
    assert lines[2].startswith("- [fact, ")
    assert lines[2].endswith("A very old note about the build")

    assert format_recall_block([]) == ""
    assert format_recall_block(None) == ""


def test_format_recall_block_relative_times():
    now = datetime.now(timezone.utc)
    record = MemoryResult(
        content="object based record",
        memory_type="fact",
        score=1.0,
        created_at=now.isoformat(),
    )
    block = format_recall_block([record])
    assert "object based record" in block
    assert "just now" in block

    recent = {"content": "recent item", "memory_type": "fact", "score": 1.0,
              "updated_at": (now - timedelta(hours=3)).isoformat()}
    assert "3h ago" in format_recall_block([recent])

    days = {"content": "days old item", "memory_type": "fact", "score": 1.0,
            "updated_at": (now - timedelta(days=4)).isoformat()}
    assert "4d ago" in format_recall_block([days])

    no_date = {"content": "no timestamp here", "memory_type": "fact", "score": 1.0}
    block = format_recall_block([no_date])
    assert "- [fact] no timestamp here" in block
