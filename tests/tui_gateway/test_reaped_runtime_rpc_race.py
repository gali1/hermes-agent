"""A background poll that loses a race with its own session teardown must not warn.

Captured (2026-11-12 16:48:24,928, logs/agent.log):

    WARNING tui_gateway.server: session-scoped RPC rejected: method=process.list
    session_id='37e80163' not in memory (detached/reaped runtime; client should resume the
    stored session), rid='r59'

The log neighbourhood shows the runtime was not a stale client. 91ms earlier the SAME client
closed it:

    16:48:24,837 INFO run_agent: OpenAI client closed (agent_close, ...) thread=MainThread
    16:48:24,928 WARNING ... session-scoped RPC rejected ... '37e80163' ...
    16:48:25,160 INFO run_agent: OpenAI client created (agent_init, ...) thread=Thread-42 (_build)
    16:57:07,883 INFO tui_gateway.server: tui prompt accepted: ui_session=6807cb9e
                  session_key=202611...bca3 agent_session_id=20261112_164824_32bca3

i.e. ``session.new`` created ``20261112_164824_32bca3`` (ui_session ``6807cb9e``) at 16:48:25 and
the client carried on. The desktop status stack polls ``process.list`` every 5s while a background
row is on screen, so the poll was simply in flight across the teardown.

Both cases still answer 4001 — that code is what the client's ``session-gone-latch`` keys on — but
only an id the gateway does *not* remember reaping is a client genuinely parked on a dead runtime.
"""

from __future__ import annotations

import logging

import pytest

from tui_gateway import server
from tui_gateway import session_lifecycle


@pytest.fixture(autouse=True)
def _isolate(_isolate_hermes_home):
    """Reap bookkeeping writes are keyed on the active home; keep it inside the sandbox."""
    session_lifecycle._reaped_session_ids.clear()
    yield
    session_lifecycle._reaped_session_ids.clear()


def _register(sid: str) -> dict:
    session = {"session_key": sid, "running": False}
    with server._sessions_lock:
        server._sessions[sid] = session
    return session


def _reject(sid: str, rid: str = "r59"):
    """Drive the real resolver and return (error_frame, warning_records)."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("tui_gateway.server")
    handler = _Collect(level=logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        session, err = server._sess_nowait({"session_id": sid}, rid)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert session is None
    return err, [r for r in records if r.levelno >= logging.WARNING]


def test_poll_racing_its_own_teardown_is_not_a_warning():
    """session.close/new pops the runtime, the in-flight poll lands 91ms later — quiet, still 4001."""
    sid = "37e80163"
    _register(sid)
    popped = server._pop_session_by_id(sid)  # the rebound copy server.py runs (split-module globals)
    assert popped is not None, "the teardown claim must pop the registered session"

    err, warnings = _reject(sid)

    assert err["error"]["code"] == 4001
    assert err["error"]["message"] == "session not found"
    assert warnings == [], f"teardown race must not warn, got {[r.getMessage() for r in warnings]}"


def test_unknown_runtime_still_warns_with_the_actionable_message():
    """An id this gateway never held is a real stale client: keep the diagnostic."""
    err, warnings = _reject("deadbeef")

    assert err["error"]["code"] == 4001
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "session-scoped RPC rejected" in message
    assert "client should resume the stored session" in message
    assert "'deadbeef'" in message


def test_reap_outside_the_memory_window_warns_again(monkeypatch):
    """The quiet window is bounded: a client still polling long after the reap is visible again."""
    sid = "cafebabe"
    _register(sid)
    server._pop_session_by_id(sid)

    monkeypatch.setattr(session_lifecycle, "_REAPED_ID_MEMORY_S", 0.0)
    assert session_lifecycle.recently_reaped_session(sid) is None

    _err, warnings = _reject(sid)
    assert len(warnings) == 1, "an expired reap entry must fall back to the warning"


def test_reap_memory_is_bounded():
    """A long-lived gateway cannot grow the map without bound."""
    for i in range(session_lifecycle._REAPED_ID_MEMORY_MAX + 50):
        session_lifecycle._note_reaped_session_id(f"sid-{i}")
    assert len(session_lifecycle._reaped_session_ids) <= session_lifecycle._REAPED_ID_MEMORY_MAX
    # The most recent entries are the ones kept.
    assert session_lifecycle.recently_reaped_session(
        f"sid-{session_lifecycle._REAPED_ID_MEMORY_MAX + 49}"
    ) is not None
