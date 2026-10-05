"""Unit tests for turn mining, secret rejection, and classification."""

from agent.memory.mining import (
    PRIVATE_BLOCK_RE,
    SECRET_PATTERNS,
    TRANSIENT_MARKERS,
    classify_candidate,
    is_secret,
    mine_turns,
    strip_private,
)


def test_secret_detection():
    assert is_secret("my key is sk-abcdefghijklmnop1234")
    assert is_secret("token ghp_abcdefghijklmnopqrst")
    assert is_secret("Authorization: Bearer abcdefghijklmnopqrst")
    assert is_secret("api_key: supersecretvalue")
    assert is_secret("password = hunter2hunter2")
    assert not is_secret("the quick brown fox jumps")
    assert not is_secret(None)
    assert not is_secret("")
    assert SECRET_PATTERNS


def test_strip_private():
    stripped = strip_private("before <private>secret stuff</private> after")
    assert "secret" not in stripped
    assert "before" in stripped and "after" in stripped
    assert strip_private("no blocks here") == "no blocks here"
    assert strip_private(None) == ""
    assert strip_private("") == ""
    assert PRIVATE_BLOCK_RE.search("<PRIVATE>hidden</PRIVATE>") is not None
    assert PRIVATE_BLOCK_RE.search("<private>line one\nline two</private>") is not None


def test_classify_candidate():
    assert classify_candidate("I prefer using uv for Python environments") == "preference"
    assert classify_candidate("We decided to use SQLite for local storage") == "decision"
    assert classify_candidate("First install the package, then run the setup step") == "procedure"
    assert classify_candidate("The build crashes with a regression error") == "bug"
    assert classify_candidate("Solved the flaky deployment with a workaround") == "solution"
    assert classify_candidate("The architecture follows a layered design pattern") == "architecture"
    assert classify_candidate("Set the config flag in the env") == "configuration"
    assert classify_candidate("The service depends on the gateway and owns its own store") == "relationship"
    assert classify_candidate("The gateway runs on port 8080 in production") == "fact"

    assert classify_candidate("hi") is None
    assert classify_candidate("too short") is None
    assert classify_candidate("") is None
    assert classify_candidate(None) is None
    assert classify_candidate("...") is None
    assert classify_candidate("sk-abcdefghijklmnop1234") is None
    assert classify_candidate("Thanks so much for your help today, really appreciate it") is None


def test_transient_markers_present():
    assert "hello" in TRANSIENT_MARKERS
    assert "thanks" in TRANSIENT_MARKERS


def test_mine_turns():
    messages = [
        {"role": "user", "content": "I prefer dark mode for all my editors please"},
        {"role": "assistant", "content": "Sure!"},
        {"role": "system", "content": "I prefer to ignore system messages entirely"},
        {"role": "user", "content": [{"type": "text", "text": "We decided to use SQLite for the session store"}]},
        {"role": "user", "content": "my key is sk-abcdefghijklmnop1234"},
        {"role": "user", "content": "The deployment pipeline uses a blue green workflow process"},
        {"role": "assistant", "content": "The architecture follows a layered design pattern"},
        {"role": "user", "content": "I prefer dark mode for all my editors please"},
    ]
    mined = mine_turns(messages)
    contents = [m["content"] for m in mined]
    types = {m["content"]: m["memory_type"] for m in mined}

    assert "I prefer dark mode for all my editors please" in contents
    assert "We decided to use SQLite for the session store" in contents
    assert types["I prefer dark mode for all my editors please"] == "preference"
    assert types["We decided to use SQLite for the session store"] == "decision"
    assert types["The deployment pipeline uses a blue green workflow process"] == "procedure"
    assert types["The architecture follows a layered design pattern"] == "architecture"
    assert all("sk-" not in content for content in contents)
    assert all("system messages" not in content for content in contents)
    assert len(contents) == len(set(contents))
    assert all(set(m) == {"content", "memory_type", "role"} for m in mined)
    assert {m["role"] for m in mined} <= {"user", "assistant"}
    assert any(m["role"] == "assistant" for m in mined)


def test_mine_turns_strips_private_and_context():
    messages = [
        {"role": "user", "content": "Remember this <private>hidden note</private> and keep the rest"},
        {"role": "user", "content": "<mempalace_context>old recall</mempalace_context> The parser uses a recursive descent design pattern"},
        {"role": "user", "content": "<rekal-context>stale</rekal-context> The session store keeps a durable write ahead log"},
        {"role": "user", "content": "<memory-context>stale</memory-context> The gateway owns the websocket transport layer"},
    ]
    mined = mine_turns(messages)
    contents = [m["content"] for m in mined]
    assert any("hidden note" not in content for content in contents)
    assert all("mempalace_context" not in content for content in contents)
    assert all("rekal-context" not in content for content in contents)
    assert all("memory-context" not in content for content in contents)
    assert all("old recall" not in content for content in contents)


def test_mine_turns_limits_and_empty():
    messages = [{"role": "user", "content": f"The service depends on component number {i} for routing"} for i in range(10)]
    assert len(mine_turns(messages, limit=3)) == 3
    assert mine_turns([], limit=3) == []
    assert mine_turns(None) == []
    assert mine_turns(messages, limit=0) == []
