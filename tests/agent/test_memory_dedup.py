"""Unit tests for the ported Hindsight text-similarity primitives.

Pure-function tests: no database, no network, no embedding backend.  Ported
from the upstream ``tests/plugins/memory/test_rekal_hindsight.py`` checks plus
new coverage for ``normalize_content`` and ``is_duplicate``.
"""

from agent.memory.dedup import (
    is_degenerate,
    is_duplicate,
    normalize_content,
    trigram_similarity,
)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


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


def test_normalize_content():
    assert normalize_content("  Hello   World  ") == "hello world"
    assert normalize_content("A\tB\nC") == "a b c"
    assert normalize_content("") == ""
    assert normalize_content(None) == ""
    assert normalize_content(42) == ""


def test_is_duplicate():
    assert is_duplicate("Hello world", "hello   WORLD")
    assert is_duplicate("the quick brown fox", "the quick brown fox!", threshold=0.9)
    assert not is_duplicate("hello world", "something else entirely")
    assert not is_duplicate("hello world", "hello there", threshold=0.99)
    assert not is_duplicate("", "")
    assert not is_duplicate(None, "something")
    assert not is_duplicate("x", None)
