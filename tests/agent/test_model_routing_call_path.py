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

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.agent.test_model_router import patched_candidates  # noqa: E402,F401 - fixture re-export


class _StopHere(Exception):
    """Sentinel: the prologue reached ``build_turn_context``."""


def _boom(*_a, **_k):
    """``build_turn_context`` replacement: stop the turn at the prologue."""
    raise _StopHere


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
    """A parts list reaches the router as its *text* (contract C9a).

    The suite used to pin ``str(list)`` here, which is what the loop did before
    C9a: the repr kept the text and dropped the attachments, so an image turn
    could be routed onto a model that cannot read images. The contract now
    splits the message with ``message_text_and_attachments`` and the text half
    is exactly the text. Pinning ``"hi"`` also fails loudly if anyone
    reintroduces the ``str()`` coercion.
    """
    _run_turn(_agent(), message=[{"type": "text", "text": "hi"}])

    assert len(prologue) == 1
    assert prologue[0] == ("test-model", "hi")


def test_a_parts_list_hands_its_modality_to_the_router(monkeypatch):
    """C9a: the attachment hint must cross the call boundary with the text."""
    import agent.conversation_loop as loop
    from agent import model_router

    seen: dict = {}

    def _spy(agent, message, **kwargs):
        seen["message"] = message
        seen["attachments"] = kwargs.get("attachments")

    monkeypatch.setattr(model_router, "route_turn", _spy)
    monkeypatch.setattr(loop, "build_turn_context", _boom)

    _run_turn(
        _agent(),
        message=[
            {"type": "text", "text": "what is in this screenshot?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    )

    assert seen["message"] == "what is in this screenshot?"
    assert seen["attachments"] == ("image",)


def test_moa_turns_are_not_routed(monkeypatch):
    """C10: the skip is delegated into ``route_turn``, not dropped at the call site.

    A skip that only exists at the call site is indistinguishable from a bug, so
    the loop now states ``moa_active=True`` and lets the router record the
    reason. The model must still be left alone — proven against the real router
    by ``test_the_real_router_leaves_a_moa_turn_on_its_model``.
    """
    import agent.conversation_loop as loop
    from agent import model_router

    seen: dict = {}

    def _spy(agent, message, **kwargs):
        seen["model"] = agent.model
        seen["moa_active"] = kwargs.get("moa_active")

    monkeypatch.setattr(model_router, "route_turn", _spy)
    monkeypatch.setattr(loop, "build_turn_context", _boom)

    _run_turn(_agent(), moa_config={"slots": []})

    assert seen == {"model": "test-model", "moa_active": True}


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


# ---------------------------------------------------------------------------
# The same two contracts, proven against the real router instead of a spy
# ---------------------------------------------------------------------------


#: A prompt that switches ``weak`` → ``strong`` under the default two-model test
#: config, so "the model did not move" is a discriminating assertion rather than
#: a vacuous one (probe: `evidence/moa-skip-falsification.txt`).
_SWITCHING_PROMPT = (
    "Analyse why this service deadlocks under load and propose an architecture "
    "change. Explain the root cause step by step and derive the trade-offs.\n"
    "```python\ndef handler():\n    pass\n```"
)


def test_the_real_router_leaves_a_moa_turn_on_its_model(patched_candidates):
    """C10 with the real engine: no switch, no scoring, a visible reason.

    The hook-level test above only proves what the loop *passes*. This one runs
    the actual ``route_turn`` with routing enabled and two candidates, so a
    future refactor that moves the guard back to the call site — or drops it —
    fails here even if the spy test still passes.
    """
    from agent import model_router as mr
    from tests.agent.test_model_router import _StubAgent, _route_config

    cfg = _route_config()  # enabled, two candidates: weak + strong
    control = _StubAgent(model="weak")
    assert mr.route_turn(control, _SWITCHING_PROMPT, config=cfg) is not None
    assert control.model == "strong"  # the same turn routes when MoA is off

    agent = _StubAgent(model="weak")
    decision = mr.route_turn(agent, _SWITCHING_PROMPT, config=cfg, moa_active=True)

    assert decision is None
    assert agent.model == "weak"
    assert agent.switch_calls == []
    assert getattr(agent, mr.ROUTING_SKIP_ATTR) == "moa_active"
