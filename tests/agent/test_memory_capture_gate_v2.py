"""Capture gate v2: probes, contextual requests, tool dumps/echoes, quarantine,
provenance, recall guard, audit/restore.

Every case below comes from a real store: the probe payloads, the skills
inventory, the session dump and the "copy the files mentioned earlier" turn were
all stored as memories by v1, and the "keep" cases are the durable facts and
standing rules a filter must never eat.
"""

import json

import pytest

import agent.memory.mining as mining
from agent.memory.backend import EnhancedMemoryBackend
from agent.memory.mining import (
    CAPTURE_FILTER_VERSION,
    assess_candidate,
    classify_candidate,
    configure_capture,
    is_request,
    looks_like_probe,
    looks_like_tool_dump,
    mine_turns,
    text_shingles,
)
from agent.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _reset_operator_patterns():
    configure_capture([], [])
    yield
    configure_capture([], [])


def _verdict(text, role="user", **kwargs):
    return assess_candidate(text, role=role, record=False, **kwargs)


# -- probes ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Reply with exactly: MEMOK",
    "Reply with exactly: WARNOK",
    "Reply with the single word: ok",
    "ok now respond with only 'PONG'",
    "Hank here, from the other Hermes box (192.168.1.210). Testing the relay "
    "path - reply with exactly 'relay-7f3a'",
    "DONEALL FINALOK READY",
    "are you there? testing the gateway",
])
def test_probes_are_dropped(text):
    assert looks_like_probe(text)
    assert _verdict(text)["reason"] == "probe"


@pytest.mark.parametrize("text", [
    "Reply with exactly one sentence when answering on Telegram",
    "Always reply with only the final answer, no preamble",
    "Respond with just the diff from now on",
])
def test_formatting_rules_shaped_like_probes_are_kept(text):
    assert not looks_like_probe(text)
    assert _verdict(text)["verdict"] == "keep"


# -- requests -------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "just copy the files mentioned earlier into the location only",
    "Use web_search for the query 'searxng latest release' and summarize",
    "ok please list all skills available",
    "can you check the gateway logs for the last hour",
    "now move those into the archive folder again",
])
def test_contextual_requests_are_dropped(text):
    assert is_request(text)
    assert _verdict(text)["reason"] == "request"


@pytest.mark.parametrize("text", [
    "Media library sync DISABLED: never auto-run media-library-sync.py --quick unless explicitly asked.",
    "Run tests with -x always before pushing",
    "Can you always answer in English from now on?",
    "remove the schedule fleet maintenance.",
    "Use uv for Python environments on this machine",
    "Set the config flag in the env",
    "Plex 32400 is LAN-only via iptables",
    "subtitle is canonical — rename video to match it",
    "cron PATH lacks ~/.local/bin so scripts must use absolute paths",
])
def test_durable_statements_are_kept(text):
    assert _verdict(text)["verdict"] == "keep", _verdict(text)


# -- tool dumps -----------------------------------------------------------------

def test_count_and_listing_headers_are_quarantined():
    for text in (
        "Last 10 sessions (173 total in state.db): 1. 2026-11-12 plex 2. 2026-11-11 esxi",
        "84 skills across 15 categories: autonomous-ai-agents claude-code - Delegate coding",
        "Oldest entry in the memory store (C:\\Users\\me\\memories.db): created_at 2026-10-16",
    ):
        verdict = _verdict(text, role="assistant")
        assert verdict == {**verdict, "verdict": "quarantine", "reason": "tool_dump"}


def test_shape_is_read_from_raw_text_not_collapsed_text():
    raw = "| id | status |\n|----|--------|\n| a1 | ok |\n| b2 | ok |\n| c3 | failed |"
    collapsed = " ".join(raw.split())
    assert looks_like_tool_dump(collapsed, raw)
    assert _verdict(collapsed, role="assistant", raw=raw)["reason"] == "tool_dump"


def test_box_drawing_render_is_quarantined():
    raw = "╭─ Hermes ──────────╮\nOldest entry: x\n╰───────────────────╯"
    assert looks_like_tool_dump(" ".join(raw.split()), raw)


def test_collapsed_inventory_is_quarantined():
    text = " ".join(f"skill-{i} \u2014 does thing {i}" for i in range(10))
    assert looks_like_tool_dump(text)


@pytest.mark.parametrize("text", [
    "3 nodes in the cluster run Debian 13 and the fourth runs Proxmox",
    "The retention pass archives rather than deletes; recall filters status active.",
])
def test_facts_with_counts_are_not_dumps(text):
    assert not looks_like_tool_dump(text)


# -- tool echo ------------------------------------------------------------------

def test_assistant_restating_tool_output_is_quarantined():
    tool = ("created_at 2026-10-16 07:40:18 id 6e36a7ed0c50470e type preference content "
            "an 8.6 KB dump of 84 skills across 15 categories autonomous agents")
    reply = ("The oldest row was created_at 2026-10-16 07:40:18 id 6e36a7ed0c50470e type "
             "preference content an 8.6 KB dump of 84 skills across 15 categories")
    shingles = text_shingles(tool)
    assert _verdict(reply, role="assistant", tool_shingles=shingles)["reason"] == "tool_echo"
    # The user quoting tool output is the user's statement; echo is assistant-only.
    assert _verdict(reply, role="user", tool_shingles=shingles)["reason"] != "tool_echo"


def test_mine_turns_reads_tool_messages_and_limits_kept_rows_only():
    tool_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    messages = [
        {"role": "user", "content": "The backup host keeps 14 daily snapshots on the NAS"},
        {"role": "tool", "content": tool_text},
        {"role": "assistant", "content": "The scan returned " + tool_text},
        {"role": "assistant", "content": "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"},
    ]
    kept = mine_turns(messages, limit=1)
    assert [m["content"] for m in kept] == [
        "The backup host keeps 14 daily snapshots on the NAS"
    ]
    assert all(set(m) == {"content", "memory_type", "role"} for m in kept)

    everything = mine_turns(messages, limit=5, include_quarantined=True)
    reasons = {m.get("quarantine") for m in everything}
    assert {"tool_echo", "tool_dump"} <= reasons


# -- operator patterns ----------------------------------------------------------

def test_operator_patterns_drop_and_keep():
    assert configure_capture([r"^status report\b"], [r"^remember:"]) == []
    assert _verdict("Status report for the nightly run, nothing new")["reason"] == "operator_pattern"
    # keep wins over a heuristic drop ...
    assert _verdict("remember: list all skills available")["verdict"] == "keep"
    # ... never over a safety drop.
    assert _verdict("remember: api_key = sk-abcdefghijklmnop1234")["reason"] == "secret"


def test_invalid_operator_pattern_is_reported_not_raised():
    errors = configure_capture(["(unclosed"], [])
    assert len(errors) == 1
    assert _verdict("Plex 32400 is LAN-only via iptables")["verdict"] == "keep"


def test_classify_candidate_stays_compatible():
    assert classify_candidate("I prefer using uv for Python environments") == "preference"
    assert classify_candidate("Reply with exactly: MEMOK") is None
    # Quarantine is "do not store as active" for legacy callers.
    assert classify_candidate("Last 10 sessions (173 total in state.db): a, b") is None
    assert mining.recent_drops(1)


# -- store ----------------------------------------------------------------------

def test_quarantined_rows_are_stored_inactive_and_never_searched(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    active = store.store("The NAS exports media over NFS v4.1 on port 2049")
    held = store.store("Last 10 sessions (173 total in state.db): NAS export x",
                       status="quarantined")
    assert held["status"] == "quarantined"
    again = store.store("Last 10 sessions (173 total in state.db): NAS export x",
                        status="quarantined")
    assert again["duplicate"] and again["memory_id"] == held["memory_id"]
    ids = {r["id"] for r in store.search("NAS export", limit=10)["results"]}
    assert active["memory_id"] in ids and held["memory_id"] not in ids
    assert store.has_content("Last 10 sessions (173 total in state.db): NAS export x") is False
    assert store.has_content("Last 10 sessions (173 total in state.db): NAS export x",
                             include_inactive=True) is True
    assert store.status_counts() == {"active": 1, "quarantined": 1}
    store.close()


def test_graph_expansion_never_surfaces_inactive_rows(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    seed = store.store("The gateway owns the websocket transport for Telegram")
    other = store.store("Websocket reconnect backoff is capped at thirty seconds")
    store.link(seed["memory_id"], other["memory_id"], "related_to")
    store.set_capture_status({other["memory_id"]: {"reason": "test"}}, status="quarantined")
    results = store.search("gateway telegram", limit=10, graph_expand=True)["results"]
    assert other["memory_id"] not in {r["id"] for r in results}
    store.close()


def test_set_capture_status_records_provenance_and_respects_expectation(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    row = store.store("Reply with exactly: MEMOK and nothing more please")
    changed = store.set_capture_status(
        {row["memory_id"]: {"reason": "probe"}}, status="quarantined", expect_status="active")
    assert changed == 1
    meta = store.get(row["memory_id"], track_access=False)["metadata"]["capture"]
    assert meta["reason"] == "probe" and meta["previous_status"] == "active"
    # Second move with a stale expectation is a no-op.
    assert store.set_capture_status(
        {row["memory_id"]: {"reason": "probe"}}, status="quarantined", expect_status="active") == 0
    store.close()


# -- backend --------------------------------------------------------------------

def _backend(tmp_path, **capture):
    config = {"enabled": True, "max_recall": 5,
              "capture": {"audit_interval_hours": 0, **capture}}
    backend = EnhancedMemoryBackend(str(tmp_path), config)
    backend.initialize("s1", platform="cli")
    return backend


def test_rows_record_role_and_filter_version(tmp_path):
    backend = _backend(tmp_path)
    backend.observe_turn("The media server is pinned to kernel 6.8 for the iGPU",
                         "Understood.", session_id="s1")
    row = backend.store.capture_rows()[0]
    assert row["metadata"]["capture"] == {"role": "user", "filter_version": CAPTURE_FILTER_VERSION}
    assert row["source"]["role"] == "user"
    backend.shutdown()


def test_observe_turn_quarantines_echo_of_this_turns_tool_output(tmp_path):
    backend = _backend(tmp_path)
    tool_text = ("created_at 2026-10-16 07:40:18 id 6e36a7ed0c50470e type preference "
                 "content raw skills_list output not a durable fact at all")
    reply = "Oldest row: " + tool_text
    messages = [
        {"role": "user", "content": "earlier question"},
        {"role": "user", "content": "List your oldest memory"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": tool_text},
        {"role": "assistant", "content": reply},
    ]
    backend.observe_turn("List your oldest memory", reply, session_id="s1", messages=messages)
    assert backend.store.capture_rows() == []
    held = backend.quarantined()
    assert held and held[0]["reason"] in {"tool_echo", "tool_dump"}
    assert "Oldest row" not in backend.prefetch("oldest row skills_list")
    backend.shutdown()


def test_assistant_capture_off_mines_user_turns_only(tmp_path):
    backend = _backend(tmp_path, assistant="off")
    backend.observe_turn("Recordings live on the second ZFS pool named tank2",
                         "The scheduler retries failed jobs three times with backoff",
                         session_id="s1")
    contents = [r["content"] for r in backend.store.capture_rows()]
    assert contents == ["Recordings live on the second ZFS pool named tank2"]
    backend.shutdown()


def test_recall_guard_hides_legacy_probe_rows_but_not_deliberate_ones(tmp_path):
    backend = _backend(tmp_path)
    store = backend.store
    # Legacy v1 rows: no capture metadata, conversation source.
    store.store("Reply with exactly: MEMOK", source={"type": "conversation"})
    store.store("Gateway restart notes: MEMOK probe passes after restart",
                source={"type": "conversation"})
    # A deliberate memory-tool write is never second-guessed.
    store.store("Reply with exactly: ACK", source={"type": "memory"})
    block = backend.prefetch("MEMOK ACK reply")
    assert "Reply with exactly: MEMOK" not in block
    assert "Gateway restart notes" in block
    assert "Reply with exactly: ACK" in block
    backend.shutdown()


def test_audit_shadow_then_apply_then_restore(tmp_path):
    backend = _backend(tmp_path)
    store = backend.store
    probe = store.store("Reply with exactly: DONEALL", source={"type": "conversation"})
    fact = store.store("Plex 32400 is LAN-only via iptables", source={"type": "conversation"})
    tool_row = store.store("Reply with exactly: FINALOK", source={"type": "memory"})

    shadow = backend.audit()
    assert shadow["mode"] == "shadow" and shadow["by_reason"] == {"probe": 1}
    assert store.status_counts()["active"] == 3

    applied = backend.audit(apply=True)
    assert applied["quarantined"] == 1
    assert store.get(probe["memory_id"], track_access=False)["status"] == "quarantined"
    assert store.get(fact["memory_id"], track_access=False)["status"] == "active"
    assert store.get(tool_row["memory_id"], track_access=False)["status"] == "active"
    assert store.capture_state("last_audit_at")

    assert backend.restore(probe["memory_id"])["success"] is True
    assert store.get(probe["memory_id"], track_access=False)["status"] == "active"
    assert backend.restore(fact["memory_id"])["success"] is False
    backend.shutdown()


def test_scheduled_audit_runs_when_due_and_records_state(tmp_path):
    seed = MemoryStore(str(tmp_path / "memory_engine" / "memories.db"))
    seed.store("Reply with exactly: PONG", source={"type": "conversation"})
    seed.close()
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True,
                                                    "capture": {"audit_interval_hours": 24}})
    backend.initialize("s1", platform="cli")
    backend._audit_thread.join(timeout=10)
    assert backend.store.status_counts().get("quarantined") == 1
    assert backend._audit_due() is False
    backend.shutdown()


def test_health_reports_capture_telemetry(tmp_path):
    backend = _backend(tmp_path)
    backend.observe_turn("Reply with exactly: READY", "READY", session_id="s1")
    capture = backend.health()["capture"]
    assert capture["filter_version"] == CAPTURE_FILTER_VERSION
    assert capture["drops_by_reason"].get("probe", 0) >= 1
    assert "quarantined_rows" in capture
    backend.shutdown()


# -- memory tool ----------------------------------------------------------------

def test_memory_tool_audit_quarantine_restore(tmp_path, monkeypatch):
    from tools.memory_tool import memory_tool

    backend = _backend(tmp_path)
    row = backend.store.store("Reply with exactly: WARNOK", source={"type": "conversation"})

    report = json.loads(memory_tool(action="audit", backend=backend))
    assert report["mode"] == "shadow" and report["flagged"] == 1

    import tools.skill_provenance as provenance
    monkeypatch.setattr(provenance, "is_unattended_review", lambda: True)
    forced = json.loads(memory_tool(action="audit", apply=True, backend=backend))
    assert forced["mode"] == "shadow"
    monkeypatch.setattr(provenance, "is_unattended_review", lambda: False)

    applied = json.loads(memory_tool(action="audit", apply=True, backend=backend))
    assert applied["quarantined"] == 1
    listed = json.loads(memory_tool(action="quarantine", backend=backend))
    assert listed["results"][0]["id"] == row["memory_id"]
    restored = json.loads(memory_tool(action="restore", memory_id=row["memory_id"], backend=backend))
    assert restored["success"] is True
    backend.shutdown()
