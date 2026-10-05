"""Unit tests for rule-based entity extraction."""

from agent.memory.entities import ENTITY_PATTERNS, extract_entities


def _by_kind(text, limit=32):
    grouped = {}
    for entity in extract_entities(text, limit=limit):
        grouped.setdefault(entity["kind"], []).append(entity["text"])
    return grouped


def test_extract_entities_kinds():
    text = (
        "See /home/bro/app/main.py and https://example.com/docs plus a@b.com; "
        "call Foo.bar or my_func, use MemoryRecord, bump to v1.2.3, "
        "set API_KEY in `config.yaml`."
    )
    grouped = _by_kind(text)
    assert "/home/bro/app/main.py" in grouped["file_path"]
    assert "https://example.com/docs" in grouped["url"]
    assert "a@b.com" in grouped["email"]
    assert "Foo.bar" in grouped["identifier"]
    assert "my_func" in grouped["identifier"]
    assert "MemoryRecord" in grouped["camel_case"]
    assert "v1.2.3" in grouped["version"]
    assert "API_KEY" in grouped["constant"]
    assert "config.yaml" in grouped["backtick"]


def test_extract_entities_dedup_and_limit():
    found = extract_entities("Foo.bar and Foo.bar and foo.bar plus my_func")
    identifiers = [e["text"] for e in found if e["kind"] == "identifier"]
    assert identifiers == ["Foo.bar", "my_func"]

    many = " ".join(f"Name{i}Thing" for i in range(100))
    assert len(extract_entities(many, limit=5)) == 5


def test_extract_entities_never_raises():
    assert extract_entities(None) == []
    assert extract_entities("") == []
    assert extract_entities("hello world") == []
    assert extract_entities("hello world", limit=0) == []
    assert extract_entities("hello world", limit=-1) == []


def test_entity_patterns_shape():
    assert ENTITY_PATTERNS
    kinds = [kind for kind, _ in ENTITY_PATTERNS]
    assert len(kinds) == len(set(kinds))
    for kind, pattern in ENTITY_PATTERNS:
        assert isinstance(kind, str) and kind
        assert hasattr(pattern, "finditer")
