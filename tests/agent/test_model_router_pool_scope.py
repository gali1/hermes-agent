"""Canonical ``smart_model_routing.models`` must honour ``preferred_models_only``.

Regression for the Section 7 item 7 defect found by the M6 audit (D-1): the
declared-models shape reported ``pool_scope = "models"``, a value neither the
strict pool nor the single-model no-op recognised.  Consequences, all fixed
here:

* with ``preferred_models_only: true`` a turn whose request needed tools or a
  modality widened to the provider catalog anyway — the exact bypass the flag
  exists to prevent;
* a one-model declared list paid the full ranking cost instead of returning
  immediately (spec §4.1: a single model means no routing overhead);
* ``route_turn`` widened while the CLI preview (``preview_decision`` /
  ``routing_report``) did not, so ``hermes model routing status`` could show a
  different answer from the one the next turn would take.

Second half of this module covers D-2: the production hook never passed
``conversation_tokens``, so a long conversation could be routed onto a model
whose context window cannot hold it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent import model_router as mr  # noqa: E402
from tests.agent.test_model_router import (  # noqa: E402
    _SPECS,
    _StubAgent,
    cand,
    patched_candidates,  # noqa: F401 - pytest fixture re-export
)


def _declared_config(models, *, preferred_only, enabled=True):
    """The canonical shape: ``smart_model_routing.models`` declares the pool."""
    return {
        "smart_model_routing": {
            "enabled": enabled,
            "preferred_models_only": preferred_only,
            "models": list(models),
        },
        # The session's provider/default still exist; they are not the pool.
        "providers": {"test": {"models": list(models)}},
        "model": {"provider": "test", "default": models[0]},
    }


VISION_PROMPT = "What is in this screenshot?"
VISION_ATTACHMENT = ("shot.png",)


# ---------------------------------------------------------------------------
# Pool scope: the flag decides, not the shape of the models list
# ---------------------------------------------------------------------------


def test_declared_models_scope_is_preferred_when_the_flag_is_on():
    cfg = _declared_config(["weak", "strong"], preferred_only=True)
    names, source, scope = mr.declared_model_list(cfg, "test", current_model="weak")
    assert source == "models"
    assert names == ["weak", "strong"]
    assert scope == "preferred"


def test_declared_models_scope_is_catalog_capable_when_the_flag_is_off():
    cfg = _declared_config(["weak", "strong"], preferred_only=False)
    _names, _source, scope = mr.declared_model_list(cfg, "test", current_model="weak")
    assert scope == "preferred+catalog"


def test_resolve_candidates_reports_the_same_scope(patched_candidates):
    strict = _declared_config(["weak", "strong"], preferred_only=True)
    wide = _declared_config(["weak", "strong"], preferred_only=False)
    assert mr.resolve_candidates(strict, "test", current_model="weak")[2] == "preferred"
    assert mr.resolve_candidates(wide, "test", current_model="weak")[2] == "preferred+catalog"


def test_legacy_candidates_list_gets_the_same_scope():
    cfg = {
        "smart_model_routing": {
            "enabled": True,
            "preferred_models_only": True,
            "candidates": ["weak", "strong"],
        },
        "model": {"provider": "test", "default": "weak"},
    }
    _names, source, scope = mr.declared_model_list(cfg, "test", current_model="weak")
    assert source == "candidates"
    assert scope == "preferred"


# ---------------------------------------------------------------------------
# The strict pool is actually strict
# ---------------------------------------------------------------------------


def test_strict_declared_pool_never_widens_for_a_modality_gap(
    patched_candidates, monkeypatch
):
    """preferred_models_only=true + models:[…] must not reach the catalog."""
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _declared_config(["weak", "strong"], preferred_only=True)
    agent = _StubAgent(model="weak")

    assert mr.route_turn(agent, VISION_PROMPT, attachments=VISION_ATTACHMENT, config=cfg) is None
    assert agent.model == "weak"
    assert agent.switch_calls == []
    assert all("vision" not in call for call in patched_candidates)


def test_strict_declared_single_model_is_a_no_op(patched_candidates):
    """One declared model: nothing to choose, so no ranking work at all."""
    cfg = _declared_config(["weak"], preferred_only=True)
    agent = _StubAgent(model="weak")
    assert mr.route_turn(agent, "explain the deadlock", config=cfg) is None
    assert agent.switch_calls == []
    assert patched_candidates == []


def test_wide_declared_pool_still_widens_for_a_modality_gap(
    patched_candidates, monkeypatch
):
    """preferred_models_only=false keeps its documented fallback."""
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _declared_config(["weak", "strong"], preferred_only=False)
    agent = _StubAgent(model="weak")

    decision = mr.route_turn(agent, VISION_PROMPT, attachments=VISION_ATTACHMENT, config=cfg)
    assert decision is not None
    assert decision.model == "vision"
    assert agent.model == "vision"
    assert "provider catalog" in decision.reason


def test_wide_declared_pool_does_not_widen_when_a_declared_model_fits(
    patched_candidates, monkeypatch
):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["tiny"])
    cfg = _declared_config(["weak", "strong"], preferred_only=False)
    agent = _StubAgent(model="weak")

    decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)
    assert decision is not None
    assert agent.model in ("weak", "strong")
    assert all("tiny" not in call for call in patched_candidates)


# ---------------------------------------------------------------------------
# Preview and runtime may not disagree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preferred_only", [True, False])
def test_preview_matches_route_turn_for_the_declared_shape(
    preferred_only, patched_candidates, monkeypatch
):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _declared_config(["weak", "strong"], preferred_only=preferred_only)

    preview, _profile, _models = mr.preview_decision(
        cfg, VISION_PROMPT, attachments=VISION_ATTACHMENT, current_model="weak"
    )
    report = mr.routing_report(
        cfg, VISION_PROMPT, attachments=VISION_ATTACHMENT, current_model="weak"
    )
    agent = _StubAgent(model="weak")
    live = mr.route_turn(agent, VISION_PROMPT, attachments=VISION_ATTACHMENT, config=cfg)

    assert (preview.model if preview else None) == (live.model if live else None)
    assert (report.get("selected") or None) == (live.model if live else None)


# ---------------------------------------------------------------------------
# D-2 — the turn hook supplies the conversation size
# ---------------------------------------------------------------------------


def _pad(messages=8, chars=40_000):
    """A conversation big enough to exceed a 2k-token window many times over."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * chars}
        for i in range(messages)
    ]


def test_route_turn_estimates_conversation_tokens_from_history(monkeypatch):
    seen = {}

    real_classify = mr.classify_request

    def _spy(prompt, **kwargs):
        seen.update(kwargs)
        return real_classify(prompt, **kwargs)

    monkeypatch.setattr(mr, "classify_request", _spy)
    cfg = _declared_config(["weak", "strong"], preferred_only=False)
    agent = _StubAgent(model="weak")

    mr.route_turn(agent, "continue", conversation_history=_pad(), config=cfg)
    assert seen.get("conversation_tokens", 0) > 10_000


def test_long_conversation_excludes_a_small_context_incumbent(
    patched_candidates, monkeypatch
):
    """``tiny`` holds a short session but must be vacated once history outgrows it."""
    cfg = _declared_config(["tiny", "weak"], preferred_only=True)

    # Short session: the 2k-window model is fine, and stability keeps it.
    short_agent = _StubAgent(model="tiny")
    short = mr.route_turn(short_agent, "what is 2+2", config=cfg)
    assert short is not None and short_agent.model == "tiny"

    # Same config, same incumbent, but the conversation alone exceeds its window:
    # context fit is a hard gate, so the session must move to the model that fits.
    agent = _StubAgent(model="tiny")
    mr.route_turn(agent, "what is 2+2", conversation_history=_pad(), config=cfg)
    assert agent.model == "weak"


def test_explicit_conversation_tokens_are_not_double_counted(
    patched_candidates, monkeypatch
):
    seen = {}
    real_classify = mr.classify_request

    def _spy(prompt, **kwargs):
        seen.update(kwargs)
        return real_classify(prompt, **kwargs)

    monkeypatch.setattr(mr, "classify_request", _spy)
    cfg = _declared_config(["weak", "strong"], preferred_only=False)

    mr.route_turn(
        _StubAgent(model="weak"),
        "continue",
        conversation_tokens=7,
        conversation_history=_pad(),
        config=cfg,
    )
    assert seen["conversation_tokens"] == 7


def test_conversation_history_is_ignored_while_routing_is_off(
    patched_candidates, monkeypatch
):
    """A disabled install must not pay for a token estimate."""
    cfg = _declared_config(["weak", "strong"], preferred_only=False, enabled=False)
    agent = _StubAgent(model="weak")

    assert mr.route_turn(agent, "continue", conversation_history=_pad(), config=cfg) is None
    assert patched_candidates == []


def test_history_estimate_never_raises():
    """A malformed history degrades to a number, never to a broken turn."""
    assert mr._conversation_token_estimate(None) == 0
    assert mr._conversation_token_estimate([]) == 0
    assert isinstance(mr._conversation_token_estimate(["not-a-message"]), int)
    assert isinstance(mr._conversation_token_estimate([{"role": "user"}]), int)
