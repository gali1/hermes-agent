"""The smart-model-routing hook, exercised through the real turn prologue.

This lineage has no ``tests/run_agent/`` fake-transport harness, so the port
pins the *wiring* contracts — the ones a refactor of the loop can silently
break — without standing up an ``AIAgent``:

1. routing is consulted exactly once per turn;
2. it runs **before** ``build_turn_context`` builds/restores the system prompt,
   so the prompt describes the model that will actually serve the turn;
3. a MoA turn and a ``codex_app_server`` turn are left alone (the router cannot
   swap the model under either);
4. a router that raises never breaks the turn.

Ranking and switching themselves are covered with real metadata in
``tests/agent/test_model_router.py`` and
``tests/agent/test_model_router_cross_provider.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class _StopHere(Exception):
    """Sentinel: the prologue reached ``build_turn_context``."""


def _agent(api_mode: str = "chat_completions"):
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "test-provider"
    agent.api_mode = api_mode
    return agent


@pytest.fixture
def prologue(monkeypatch):
    """Replace ``build_turn_context`` with a sentinel and record routing calls."""
    import agent.conversation_loop as loop
    from agent import model_router

    seen: list = []

    def _spy(agent, message, **kwargs):
        seen.append((agent.model, message))
        return None

    monkeypatch.setattr(model_router, "route_turn", _spy)

    def _boom(*_a, **_k):
        raise _StopHere

    monkeypatch.setattr(loop, "build_turn_context", _boom)
    return seen


def _run_turn(agent, message="hello there", **kwargs):
    import agent.conversation_loop as loop

    with pytest.raises(_StopHere):
        loop._run_conversation_turn(agent, message, **kwargs)


def test_router_is_consulted_once_before_the_prompt_is_built(prologue):
    _run_turn(_agent())

    assert len(prologue) == 1
    assert prologue[0] == ("test-model", "hello there")


def test_router_sees_a_non_string_message_as_text(prologue):
    """Multimodal turns hand the loop a list; classification needs text."""
    _run_turn(_agent(), message=[{"type": "text", "text": "hi"}])

    assert len(prologue) == 1
    assert prologue[0][1] == "[{'type': 'text', 'text': 'hi'}]"


def test_moa_turns_are_not_routed(prologue):
    _run_turn(_agent(), moa_config={"slots": []})

    assert prologue == []


def test_codex_app_server_turns_are_not_routed(prologue):
    _run_turn(_agent(api_mode="codex_app_server"))

    assert prologue == []


def test_a_failing_router_never_breaks_the_turn(monkeypatch):
    """Routing is an optimisation: its bugs must not surface to the user."""
    import agent.conversation_loop as loop
    from agent import model_router

    def _explode(*_a, **_k):
        raise RuntimeError("router bug")

    monkeypatch.setattr(model_router, "route_turn", _explode)
    monkeypatch.setattr(loop, "build_turn_context", lambda *a, **k: (_ for _ in ()).throw(_StopHere()))

    _run_turn(_agent())  # the sentinel still fires: the turn continued
