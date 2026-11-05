"""Regression tests for the two /model-routing defects reported from chat.

1. "no way to add or remove models" - the models step read flow["candidates"], which nothing
   ever populated, so the page said "no eligible models" and the list could never change.
   The picker must offer the pool routing COULD choose (configured + provider catalog), with
   the configured models first so a confirmed order keeps its meaning.
2. "Picker expired" on a live picker - state was stored per chat only and popped by the first
   tap, so a second picker in the same chat, an adapter swap, or a tap on an older message all
   looked expired. State now resolves by (chat, message) through a process-wide registry.
"""
from __future__ import annotations

import time

from gateway import run_busy
from plugins.platforms.telegram import adapter as tg


def _holder(cfg):
    holder = type("G", (), {})()
    holder._model_routing_load_config = staticmethod(lambda: cfg)
    return holder


def test_models_step_offers_catalog_not_just_the_configured_row(monkeypatch):
    cfg = {"model": {"provider": "opencode-zen", "default": "deepseek-v4.1-flash"}}
    settings = {"models": ["deepseek-v4.1-flash"]}

    seen = {}

    def fake_widen(provider, models, **kwargs):
        seen["provider"] = provider
        seen["models"] = list(models)
        return ["deepseek-v4.1-flash", "gpt-5-nano", "claude-haiku-5-5"]

    monkeypatch.setattr("agent.model_router.widen_pool", fake_widen)
    choices = run_busy.GatewayBusySessionMixin._model_routing_model_choices(_holder(cfg), settings)

    assert seen["provider"] == "opencode-zen"
    assert choices[0] == "deepseek-v4.1-flash", "the configured model must keep its priority"
    assert len(choices) > 1, "the picker must be able to add a model, not only re-pick one"


def test_configured_models_take_precedence_over_the_catalog(monkeypatch):
    cfg = {"model": {"provider": "opencode-zen", "default": "b-model"}}
    settings = {"models": ["a-model", "b-model"]}
    monkeypatch.setattr(
        "agent.model_router.widen_pool",
        lambda provider, models, **kw: list(models) + ["c-catalog"],
    )
    choices = run_busy.GatewayBusySessionMixin._model_routing_model_choices(_holder(cfg), settings)
    assert choices == ["a-model", "b-model", "c-catalog"]


def test_a_catalog_failure_still_offers_the_configured_models(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("catalog down")

    monkeypatch.setattr("agent.model_router.widen_pool", boom)
    cfg = {"model": {"provider": "opencode-zen", "default": "deepseek-v4.1-flash"}}
    choices = run_busy.GatewayBusySessionMixin._model_routing_model_choices(
        _holder(cfg), {"models": ["deepseek-v4.1-flash"]}
    )
    assert choices == ["deepseek-v4.1-flash"]


# ── picker registry ────────────────────────────────────────────────────────────────


def test_picker_state_is_found_by_chat_and_message():
    tg._PICKER_STATE_REGISTRY.clear()
    state = {"msg_id": 77, "choices": [{"value": "1"}]}
    tg._picker_state_register("852", 77, state)
    assert tg._picker_state_lookup("852", 77) is state
    assert tg._picker_state_lookup("852", 78) is None, "another message must not inherit it"
    assert tg._picker_state_lookup("999", 77) is None, "another chat must not inherit it"


def test_retiring_one_message_leaves_another_picker_alive():
    tg._PICKER_STATE_REGISTRY.clear()
    first = {"msg_id": 1}
    second = {"msg_id": 2}
    tg._picker_state_register("852", 1, first)
    tg._picker_state_register("852", 2, second)
    tg._picker_state_retire("852", 1)
    assert tg._picker_state_lookup("852", 1) is None
    assert tg._picker_state_lookup("852", 2) is second, "a second picker in the same chat survives"


def test_an_abandoned_picker_stops_answering(monkeypatch):
    tg._PICKER_STATE_REGISTRY.clear()
    tg._picker_state_register("852", 5, {"msg_id": 5})
    later = time.time() + tg._PICKER_STATE_TTL + 1
    monkeypatch.setattr(tg.time, "time", lambda: later)
    assert tg._picker_state_lookup("852", 5) is None


def test_current_models_come_from_the_provider_list_so_they_arrive_preselected():
    """Nothing pre-ticked means the picker can only add - removing the model in use becomes
    impossible, which is half of what was reported."""
    cfg = {"model": {"provider": "opencode-zen", "default": "deepseek-v4.1-flash"},
           "providers": {"opencode-zen": {"models": ["deepseek-v4.1-flash", "gpt-5-nano"]}}}
    holder = _holder(cfg)
    current = run_busy.GatewayBusySessionMixin._model_routing_current_models(
        holder, {"models": []}, {})
    assert current == ["deepseek-v4.1-flash", "gpt-5-nano"]


def test_current_models_fall_back_to_the_default_model():
    cfg = {"model": {"provider": "opencode-zen", "default": "deepseek-v4.1-flash"}}
    current = run_busy.GatewayBusySessionMixin._model_routing_current_models(
        _holder(cfg), {}, {})
    assert current == ["deepseek-v4.1-flash"]


def test_a_pending_change_wins_over_the_saved_list():
    cfg = {"model": {"provider": "opencode-zen", "default": "deepseek-v4.1-flash"},
           "providers": {"opencode-zen": {"models": ["deepseek-v4.1-flash"]}}}
    current = run_busy.GatewayBusySessionMixin._model_routing_current_models(
        _holder(cfg), {}, {"models": ["gpt-5-nano"]})
    assert current == ["gpt-5-nano"]
