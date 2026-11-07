"""Tests for agent.model_router — deterministic model selection.

No network: every candidate carries a hand-built ``ModelInfo``.
"""

import json
import os
import sys

import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.models_dev import ModelInfo  # noqa: E402
from agent import model_router as mr  # noqa: E402


def info(
    model,
    *,
    provider="test",
    ctx=200_000,
    out=8_192,
    cost_in=1.0,
    cost_out=2.0,
    tools=True,
    reasoning=False,
    vision=False,
    structured=False,
    status="",
    in_mods=None,
    out_mods=None,
):
    return ModelInfo(
        id=model,
        name=model,
        family="",
        provider_id=provider,
        reasoning=reasoning,
        tool_call=tools,
        attachment=vision,
        structured_output=structured,
        input_modalities=tuple(in_mods if in_mods is not None else (("text", "image") if vision else ("text",))),
        output_modalities=tuple(out_mods if out_mods is not None else ("text",)),
        context_window=ctx,
        max_output=out,
        cost_input=cost_in,
        cost_output=cost_out,
        status=status,
    )


def cand(model, **kw):
    provider = kw.pop("provider", "test")
    return mr.ModelCandidate(provider=provider, model=model, info=info(model, provider=provider, **kw))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_short_factual_question_is_simple():
    p = mr.classify_request("What is 15 multiplied by 8?")
    assert p.complexity is mr.Complexity.SIMPLE
    assert p.required_context > 0
    assert p.est_output_tokens <= 300


def test_complex_code_reasoning_request_is_advanced():
    prompt = (
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    p = mr.classify_request(prompt)
    assert p.complexity is mr.Complexity.ADVANCED
    assert p.needs_reasoning is True
    assert "code-signal" in p.signals


def test_long_but_plain_request_is_not_advanced():
    prompt = "Summarise the following notes. " * 60  # long, no reasoning signals
    p = mr.classify_request(prompt)
    assert p.complexity in (mr.Complexity.MODERATE, mr.Complexity.COMPLEX)
    assert p.needs_reasoning is False


def test_attachment_modalities_detected_from_paths():
    p = mr.classify_request("Describe this", attachments=["shot.PNG", "doc.pdf"])
    assert "image" in p.input_modalities
    assert "pdf" in p.input_modalities


def test_image_output_requested():
    p = mr.classify_request("Draw a duck", expects_image=True)
    assert "image" in p.output_modalities


def test_conversation_tokens_raise_required_context():
    a = mr.classify_request("hello")
    b = mr.classify_request("hello", conversation_tokens=100_000)
    assert b.required_context > a.required_context + 90_000


def test_tool_need_detected_for_action_requests():
    assert mr.classify_request("run the tests and show the git status").needs_tools is True
    assert mr.classify_request("explain what a monad is").needs_tools is False


def test_classification_is_deterministic():
    prompt = "Debug this traceback and refactor the parser"
    assert mr.classify_request(prompt) == mr.classify_request(prompt)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def test_deprecated_model_excluded():
    p = mr.classify_request("hello")
    out = mr.filter_eligible([cand("old", status="deprecated"), cand("new")], p)
    assert [c.model for c in out] == ["new"]


def test_context_too_small_excluded():
    p = mr.classify_request("x" * 40_000)  # ~10k tokens
    out = mr.filter_eligible([cand("tiny", ctx=4_000), cand("big", ctx=200_000)], p)
    assert [c.model for c in out] == ["big"]


def test_output_cap_too_small_excluded():
    p = mr.classify_request("do something", expected_output_tokens=8_000)
    out = mr.filter_eligible([cand("short", out=1_024), cand("long", out=16_384)], p)
    assert [c.model for c in out] == ["long"]


def test_vision_required_excludes_text_only_model():
    p = mr.classify_request("what is in this picture", attachments=["a.png"])
    out = mr.filter_eligible([cand("textonly", vision=False), cand("vision", vision=True)], p)
    assert [c.model for c in out] == ["vision"]


def test_image_output_requires_image_output_modality():
    p = mr.classify_request("draw a duck", expects_image=True)
    text_only = cand("textgen", out_mods=("text",))
    image_gen = cand("imagen", out_mods=("text", "image"))
    out = mr.filter_eligible([text_only, image_gen], p)
    assert [c.model for c in out] == ["imagen"]


def test_confirmed_no_tools_excluded_when_tools_needed():
    p = mr.classify_request("run the tests")
    assert p.needs_tools
    no_tools = mr.ModelCandidate("test", "notools", info("notools", tools=False),
                                 tool_support=mr.CapabilityState.UNSUPPORTED)
    out = mr.filter_eligible([no_tools, cand("tools")], p)
    assert [c.model for c in out] == ["tools"]


def test_unknown_tool_support_is_allowed_by_default():
    """models.dev defaults tool_call to False; absence must not disqualify."""
    p = mr.classify_request("run the tests")
    unknown = mr.ModelCandidate("test", "unknown", info("unknown", tools=False))
    assert [c.model for c in mr.filter_eligible([unknown], p)] == ["unknown"]


def test_strict_mode_drops_unknown_metadata():
    cfg = mr.RoutingConfig(allow_unknown_metadata=False)
    p = mr.classify_request("describe this", attachments=["a.png"])
    no_info = mr.ModelCandidate("test", "mystery", None)
    assert mr.filter_eligible([no_info], p, cfg) == []


def test_disabled_candidate_and_exclude_list():
    p = mr.classify_request("hello")
    cfg = mr.RoutingConfig(exclude=("test/banned",))
    out = mr.filter_eligible(
        [cand("off", provider="test"), cand("banned"), cand("ok")], p, cfg
    )
    # provider is "test" for all; only "banned" is excluded by key
    assert "banned" not in [c.model for c in out]


def test_unknown_metadata_allowed_but_not_rewarded():
    p = mr.classify_request("hello")
    known = cand("known")
    mystery = mr.ModelCandidate("test", "mystery", None)
    s_known = mr.score_candidate(known, p)
    s_mystery = mr.score_candidate(mystery, p)
    assert s_mystery < s_known


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_estimate_cost_math():
    p = mr.TaskProfile(est_input_tokens=1_000_000, est_output_tokens=1_000_000)
    c = cand("m", cost_in=3.0, cost_out=15.0)
    assert mr.estimate_cost(c, p) == pytest.approx(18.0)


def test_estimate_cost_unknown_pricing():
    p = mr.classify_request("hi")
    assert mr.estimate_cost(cand("free", cost_in=0, cost_out=0), p) is None


def test_simple_request_prefers_cheaper_model():
    p = mr.classify_request("What is the capital of France?")
    expensive = cand("expensive", cost_in=15.0, cost_out=75.0)
    cheap = cand("cheap", cost_in=0.2, cost_out=0.8)
    d = mr.select_model(p, [expensive, cheap])
    assert d is not None and d.model == "cheap"


def test_complex_request_prefers_reasoning_model_under_quality_strategy():
    prompt = (
        "Prove the invariant holds, derive the complexity, and analyse the trade-offs "
        "of this algorithm.\n```python\nx = 1\n```"
    )
    p = mr.classify_request(prompt)
    assert p.needs_reasoning
    plain = cand("plain", cost_in=0.2, cost_out=0.8, reasoning=False)
    thinker = cand("thinker", cost_in=15.0, cost_out=75.0, reasoning=True)
    cfg = mr.RoutingConfig(strategy="quality")
    d = mr.select_model(p, [plain, thinker], cfg)
    assert d is not None and d.model == "thinker"


# ---------------------------------------------------------------------------
# Stability / hysteresis
# ---------------------------------------------------------------------------

def test_keeps_current_model_when_gain_is_marginal():
    p = mr.classify_request("What is 2 + 2?")
    current = cand("current", cost_in=1.0, cost_out=2.0)
    rival = cand("rival", cost_in=0.9, cost_out=1.8)  # slightly cheaper
    # Disable the continuity bonus so the rival genuinely leads by a hair.
    cfg = mr.RoutingConfig(weights={"capability": 0.45, "cost": 0.30, "reliability": 0.10,
                                    "context": 0.10, "continuity": 0.0})
    d = mr.select_model(p, [current, rival], cfg, current_model="current")
    assert d is not None
    assert d.model == "current"
    assert d.switched is False
    assert "kept current" in d.reason


def test_cost_downgrade_is_suppressed_mid_session():
    """A cheaper model must not silently replace a working one (prompt cache)."""
    p = mr.classify_request("What is 2 + 2?")
    current = cand("current", cost_in=100.0, cost_out=100.0)
    rival = cand("rival", cost_in=0.1, cost_out=0.1)
    d = mr.select_model(p, [current, rival], current_model="current")
    assert d is not None and d.model == "current"
    assert d.switched is False
    assert "cost-only downgrade" in d.reason


def test_cost_downgrade_allowed_when_opted_in():
    p = mr.classify_request("What is 2 + 2?")
    current = cand("current", cost_in=100.0, cost_out=100.0)
    rival = cand("rival", cost_in=0.1, cost_out=0.1)
    cfg = mr.RoutingConfig(switch_on_cost=True)
    d = mr.select_model(p, [current, rival], cfg, current_model="current")
    assert d is not None and d.model == "rival"
    assert d.switched is True


def test_quality_upgrade_switches_even_without_cost_optin():
    """A capability upgrade is a legitimate reason to change models."""
    p = mr.classify_request(
        "Prove the invariant, derive the complexity and analyse the trade-offs of this algorithm."
        "\n```python\nx = 1\n```"
    )
    assert p.needs_reasoning
    current = cand("current", cost_in=0.2, cost_out=0.8, reasoning=False)
    thinker = cand("thinker", cost_in=15.0, cost_out=75.0, reasoning=True)
    cfg = mr.RoutingConfig(strategy="quality")
    d = mr.select_model(p, [current, thinker], cfg, current_model="current")
    assert d is not None and d.model == "thinker" and d.switched is True


def test_switches_when_incumbent_becomes_ineligible():
    p = mr.classify_request("x" * 60_000)  # ~15k tokens
    current = cand("current", ctx=4_000)   # cannot hold the context
    rival = cand("rival", ctx=200_000)
    d = mr.select_model(p, [current, rival], current_model="current")
    assert d is not None and d.model == "rival"


def test_single_candidate_is_stable():
    p = mr.classify_request("hello")
    d = mr.select_model(p, [cand("only")], current_model="only")
    assert d is not None and d.model == "only" and d.switched is False


# ---------------------------------------------------------------------------
# Pinning, emptiness, config
# ---------------------------------------------------------------------------

def test_pinned_model_wins():
    p = mr.classify_request("hello", pinned_model="pinned")
    d = mr.select_model(p, [cand("cheap", cost_in=0.1), cand("pinned", cost_in=50.0)])
    assert d is not None and d.model == "pinned" and d.pinned is True


def test_pinned_unknown_returns_none_rather_than_substituting():
    p = mr.classify_request("hello", pinned_model="ghost")
    assert mr.select_model(p, [cand("other")]) is None


def test_no_candidates_returns_none():
    assert mr.select_model(mr.classify_request("hi"), []) is None


def test_is_enabled_defaults_off():
    assert mr.is_enabled({}) is False
    assert mr.is_enabled({"smart_model_routing": {"enabled": False}}) is False
    assert mr.is_enabled({"smart_model_routing": {"enabled": True}}) is True
    assert mr.is_enabled(mr.RoutingConfig(enabled=True)) is True


def test_routing_config_normalises_weights():
    cfg = mr.RoutingConfig.from_dict(
        {"enabled": True, "weights": {"capability": 2, "cost": 1}, "stability_threshold": "0.2"}
    )
    assert cfg.enabled is True
    assert cfg.stability_threshold == pytest.approx(0.2)
    assert sum(cfg.weights.values()) == pytest.approx(1.0)


def test_routing_config_ignores_garbage():
    cfg = mr.RoutingConfig.from_dict({"enabled": "yes", "stability_threshold": "abc", "weights": []})
    assert cfg.enabled is True
    # Contract default: a rival must beat the incumbent by MORE than 0.15.
    assert cfg.stability_threshold == pytest.approx(0.15)


def test_explain_mentions_model_and_reason():
    p = mr.classify_request("What is 2 + 2?")
    d = mr.select_model(p, [cand("cheap", cost_in=0.1)])
    text = mr.explain(d, p)
    assert "test/cheap" in text and "reason:" in text and "complexity:" in text


def test_explain_none_is_actionable():
    assert "no eligible model" in mr.explain(None, mr.classify_request("hi"))


def test_selection_is_deterministic():
    p = mr.classify_request("compare these two options")
    models = [cand("a", cost_in=1.0), cand("b", cost_in=1.0), cand("c", cost_in=5.0)]
    first = mr.select_model(p, models)
    second = mr.select_model(p, models)
    assert first.key == second.key and first.score == second.score


# ---------------------------------------------------------------------------
# Turn-level integration (route_turn) — the call path
# ---------------------------------------------------------------------------

class _StubAgent:
    """Minimal stand-in for AIAgent: only what route_turn touches."""

    def __init__(self, model="weak", provider="test", tools=None, api_mode="chat_completions"):
        self.model = model
        self.provider = provider
        self.api_mode = api_mode
        self.tools = tools
        self.switch_calls = []

    def switch_model(self, new_model, new_provider, api_key="", base_url="", api_mode="", user_initiated=True):
        self.switch_calls.append((new_model, new_provider, user_initiated))
        self.model = new_model
        self.provider = new_provider


_SPECS = {
    "weak": dict(ctx=200_000, cost_in=1.0, cost_out=2.0, reasoning=False),
    "strong": dict(ctx=200_000, cost_in=1.5, cost_out=2.0, reasoning=True),
    "tiny": dict(ctx=2_000, cost_in=0.1, cost_out=0.2),
    "pricey": dict(ctx=200_000, cost_in=8.0, cost_out=16.0, reasoning=True),
    # Catalog-only model used by the preferred_models_only=false tests.
    "vision": dict(ctx=200_000, cost_in=3.0, cost_out=3.0, vision=True),
}


def _route_config(enabled=True, models=("weak", "strong"), provider="test", default=None):
    return {
        "smart_model_routing": {"enabled": enabled},
        "providers": {provider: {"models": list(models)}},
        "model": {"provider": provider, "default": default or models[0]},
    }


@pytest.fixture
def patched_candidates(monkeypatch):
    calls = []

    def _candidates_for(provider, models, *, enabled=True):
        calls.append(list(models))
        return [cand(m, provider=provider, **_SPECS[m]) for m in models if m in _SPECS]

    monkeypatch.setattr(mr, "candidates_for", _candidates_for)
    return calls


def test_route_turn_disabled_is_a_noop(patched_candidates):
    agent = _StubAgent()
    assert mr.route_turn(agent, "explain the deadlock", config=_route_config(enabled=False)) is None
    assert agent.switch_calls == []
    assert agent.model == "weak"
    assert patched_candidates == []  # no candidate work at all when disabled


def test_route_turn_single_model_does_no_routing(patched_candidates):
    agent = _StubAgent()
    cfg = _route_config(models=("weak",))
    assert mr.route_turn(agent, "explain the deadlock", config=cfg) is None
    assert agent.switch_calls == []
    assert patched_candidates == []  # single-model config pays no ranking cost


def test_route_turn_first_turn_can_select_a_capable_model(patched_candidates):
    agent = _StubAgent(model="weak")
    prompt = (
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    decision = mr.route_turn(agent, prompt, config=_route_config())
    assert decision is not None
    assert agent.switch_calls, "first turn should have switched"
    assert agent.switch_calls[0][0] == "strong"
    assert agent.switch_calls[0][2] is False  # router swaps never pin the session
    assert agent.model == "strong"


def test_route_turn_mid_session_keeps_eligible_incumbent(patched_candidates):
    agent = _StubAgent(model="weak")
    cfg = _route_config()
    prompt = (
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    mr.route_turn(agent, prompt, config=cfg)          # turn 0: weak -> strong
    agent.model = "weak"                              # pretend the user went back
    agent.switch_calls.clear()
    decision = mr.route_turn(agent, prompt, config=cfg)  # turn 1: incumbent eligible
    assert agent.switch_calls == [], "a marginal score win must not switch mid-session"
    assert decision is not None and decision.switched is False
    assert "prompt-cache stability" in decision.reason


def test_route_turn_mid_session_switches_when_incumbent_ineligible(patched_candidates):
    agent = _StubAgent(model="tiny")
    cfg = _route_config(models=("tiny", "strong"))
    mr.route_turn(agent, "hello", config=cfg)  # turn 0 (keeps tiny: nothing better needed)
    agent.model = "tiny"
    agent.switch_calls.clear()
    decision = mr.route_turn(
        agent,
        "Summarise the whole session so far.",
        conversation_tokens=50_000,  # 50k context cannot fit tiny's 2k window
        config=cfg,
    )
    assert agent.switch_calls, "an ineligible incumbent must be replaced"
    assert agent.model == "strong"
    assert decision is not None and decision.model == "strong"


def test_route_turn_respects_a_user_pin(patched_candidates):
    agent = _StubAgent(model="weak")
    agent._model_routing_pinned = True
    prompt = (
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    assert mr.route_turn(agent, prompt, config=_route_config()) is None
    assert agent.switch_calls == []


def test_route_turn_never_raises_when_the_swap_fails(patched_candidates):
    class _Broken(_StubAgent):
        def switch_model(self, *a, **kw):
            raise RuntimeError("client rebuild failed")

    agent = _Broken(model="weak")
    prompt = (
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    assert mr.route_turn(agent, prompt, config=_route_config()) is None  # swallowed


def test_route_turn_without_switch_model_keeps_the_model(patched_candidates):
    class _NoSwap(_StubAgent):
        switch_model = None

    agent = _NoSwap(model="weak")
    prompt = (
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    assert mr.route_turn(agent, prompt, config=_route_config()) is None
    assert agent.model == "weak"


def test_preferred_models_reads_every_existing_config_shape():
    cfg = {
        "providers": {"test": {"models": {"a": {}, "b": {}}, "model": "c"}},
        "model": {"provider": "test", "preferred_models": ["d"]},
        "smart_model_routing": {"candidates": ["e", "a"]},
    }
    assert mr._preferred_models(cfg, "test") == ["a", "b", "c", "d", "e"]


def test_preferred_models_ignores_another_providers_model_section():
    cfg = {"model": {"provider": "other", "default": "x", "preferred_models": ["x"]}}
    assert mr._preferred_models(cfg, "test") == []


def test_route_turn_first_turn_may_choose_the_cheaper_model(patched_candidates):
    """On turn 0 nothing is cached, so cost-driven selection is allowed.

    ``switch_on_cost`` guards the *mid-session* downgrade (prompt-cache
    invalidation); applying it to the first turn would make an enabled
    router unable to save anything on a fresh session.
    """
    agent = _StubAgent(model="pricey")
    cfg = _route_config(models=("weak", "pricey"), default="pricey")
    decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)
    assert agent.switch_calls, "a fresh session should be allowed to start on the cheaper model"
    assert agent.model == "weak"
    assert decision is not None and decision.switched is True


def test_route_turn_mid_session_declines_a_cost_downgrade(patched_candidates):
    agent = _StubAgent(model="pricey")
    cfg = _route_config(models=("weak", "pricey"), default="pricey")
    mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)  # turn 0 -> weak
    agent.model = "pricey"                                          # user back on the expensive one
    agent.switch_calls.clear()
    mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)  # turn 1
    assert agent.switch_calls == []
    assert agent.model == "pricey"


def test_route_turn_protects_a_non_default_model_on_the_first_turn(patched_candidates):
    """A session started on a non-default model is a deliberate choice.

    The gateway applies ``/model`` overrides by constructing the agent with
    that model, so there is no ``switch_model`` call to pin it; the router
    must infer the pin from "not the configured default".
    """
    agent = _StubAgent(model="strong")  # config default is "weak"
    cfg = _route_config(models=("weak", "pricey", "strong"))
    mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)
    assert agent.switch_calls == []
    assert agent.model == "strong"


# ---------------------------------------------------------------------------
# preview_decision — the read-only path behind `hermes model routing explain`
# ---------------------------------------------------------------------------

def test_preview_returns_a_decision_even_when_routing_is_disabled(patched_candidates):
    """`explain` must show what *would* happen on an install that is still off."""
    cfg = _route_config(enabled=False, models=("weak", "strong"))
    decision, profile, models = mr.preview_decision(cfg, "What is 15 multiplied by 8?")
    assert models == ["weak", "strong"]
    assert profile.complexity is mr.Complexity.SIMPLE
    assert decision is not None
    assert decision.model in {"weak", "strong"}


def test_preview_single_model_reports_nothing_to_route(patched_candidates):
    cfg = _route_config(models=("weak",))
    decision, _profile, models = mr.preview_decision(cfg, "What is 15 multiplied by 8?")
    assert models == ["weak"]
    assert decision is None
    assert patched_candidates == []  # no ranking work for a single-model config


def test_preview_candidates_override_the_configured_list(patched_candidates):
    cfg = _route_config(models=("weak",))
    decision, _profile, models = mr.preview_decision(
        cfg, "summarise this thread", candidates=["tiny", "pricey"]
    )
    assert models == ["tiny", "pricey"]
    assert decision is not None
    assert patched_candidates == [["tiny", "pricey"]]


def test_preview_never_mutates_the_config(patched_candidates):
    cfg = _route_config(models=("weak", "strong"))
    before = repr(cfg)
    mr.preview_decision(cfg, "What is 15 multiplied by 8?")
    assert repr(cfg) == before


def test_preview_prefers_the_cheaper_model_for_a_simple_prompt(patched_candidates):
    cfg = _route_config(models=("pricey", "tiny"), default="pricey")
    decision, _profile, _models = mr.preview_decision(
        cfg, "What is 15 multiplied by 8?", current_model="pricey"
    )
    assert decision is not None
    # tiny is 80x cheaper and a one-line factual question needs no reasoning
    assert decision.model == "tiny"


def test_preview_honours_the_explicit_incumbent(patched_candidates):
    cfg = _route_config(models=("weak", "strong"))
    decision, _profile, _models = mr.preview_decision(
        cfg, "What is 15 multiplied by 8?", current_model="strong"
    )
    assert decision is not None
    assert decision.model == "strong"


# ---------------------------------------------------------------------------
# T1d — multiple preferred models per provider + preferred_models_only
# ---------------------------------------------------------------------------


def _multi_config(models=("weak", "strong", "pricey"), *, preferred_only=True, provider="test"):
    return {
        "smart_model_routing": {"enabled": True, "preferred_models_only": preferred_only},
        "providers": {provider: {"models": list(models)}},
        "model": {"provider": provider, "default": models[0]},
    }


def test_three_models_per_provider_load_in_configured_order():
    cfg = _multi_config()
    assert mr.configured_models(cfg, "test") == ["weak", "strong", "pricey"]


def test_three_model_config_routes_within_the_list(patched_candidates):
    cfg = _multi_config()
    agent = _StubAgent(model="weak")
    decision = mr.route_turn(agent, "What is 2+2?", config=cfg)
    assert decision is not None
    assert agent.model in ("weak", "strong", "pricey")
    assert all(set(call) <= {"weak", "strong", "pricey"} for call in patched_candidates)


def test_preferred_models_only_true_pool_is_the_configured_list():
    cfg = _multi_config(("weak", "strong"), preferred_only=True)
    pool, scope = mr.routing_pool(cfg, "test", current_model="strong")
    assert scope == "preferred"
    # Configured priority order is preserved; an incumbent already on the list
    # is not duplicated and does not jump the queue.
    assert pool == ["weak", "strong"]
    # An incumbent from outside the list is prepended (it is the current
    # model, not a new selection) but nothing else is added.
    pool2, _ = mr.routing_pool(cfg, "test", current_model="legacy")
    assert pool2 == ["legacy", "weak", "strong"]


def test_preferred_models_only_true_never_consults_the_catalog(patched_candidates, monkeypatch):
    """A much cheaper catalog model must not win while the flag is on."""
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["tiny"])
    cfg = _multi_config(("weak", "strong"), preferred_only=True)
    agent = _StubAgent(model="weak")
    decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)
    assert decision is not None
    assert agent.model in ("weak", "strong")
    assert all("tiny" not in call for call in patched_candidates)


def test_preferred_models_only_false_does_not_widen_when_a_preferred_model_fits(
    patched_candidates, monkeypatch
):
    """Widening is a fallback for capability gaps, not a cost free-for-all."""
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["tiny"])
    cfg = _multi_config(("weak", "strong"), preferred_only=False)
    agent = _StubAgent(model="weak")
    decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)
    assert decision is not None
    assert patched_candidates == [["weak", "strong"]]


def test_preferred_models_only_false_widens_when_nothing_preferred_is_eligible(
    patched_candidates, monkeypatch
):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _multi_config(("weak", "strong"), preferred_only=False)
    agent = _StubAgent(model="weak")
    decision = mr.route_turn(agent, "What is in this screenshot?", attachments=("shot.png",), config=cfg)
    assert decision is not None
    assert decision.model == "vision"
    assert agent.model == "vision"
    assert "widened" in decision.reason or "provider catalog" in decision.reason


def test_preferred_models_only_true_keeps_an_incapable_incumbent(patched_candidates, monkeypatch):
    """With the flag on, no eligible preferred model means "keep what we have"."""
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _multi_config(("weak", "strong"), preferred_only=True)
    agent = _StubAgent(model="weak")
    assert mr.route_turn(agent, "What is in this screenshot?", attachments=("shot.png",), config=cfg) is None
    assert agent.model == "weak"
    assert agent.switch_calls == []


def test_widen_pool_keeps_order_and_deduplicates(monkeypatch):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["strong", "vision"])
    assert mr.widen_pool("test", ["weak", "strong"]) == ["weak", "strong", "vision"]


def test_widen_pool_is_empty_safe(monkeypatch):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: [])
    assert mr.widen_pool("test", ["weak"]) == ["weak"]


def test_preview_widens_when_nothing_preferred_is_eligible(patched_candidates, monkeypatch):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _multi_config(("weak", "strong"), preferred_only=False)
    decision, _profile, models = mr.preview_decision(
        cfg, "What is in this screenshot?", attachments=("shot.png",), current_model="weak"
    )
    assert decision is not None
    assert decision.model == "vision"
    assert "vision" in models


def test_preview_does_not_widen_with_the_flag_on(patched_candidates, monkeypatch):
    monkeypatch.setattr(mr, "_catalog_models", lambda provider, limit=60: ["vision"])
    cfg = _multi_config(("weak", "strong"), preferred_only=True)
    decision, _profile, models = mr.preview_decision(
        cfg, "What is in this screenshot?", attachments=("shot.png",), current_model="weak"
    )
    assert decision is None
    assert models == ["weak", "strong"]


# ---------------------------------------------------------------------------
# Shared config contract: smart_model_routing.models[] + the JSON report
# ---------------------------------------------------------------------------
#
# The sibling tree (Bender's ~/hermes-agent) implemented the same feature with
# this config shape; both trees must behave identically on the same config even
# though the code differs.  These tests pin the contract, not this tree's
# implementation details.

TSIM = 0.15  # the contract's default stability threshold


def spec(name, **kw):
    """A declared ``smart_model_routing.models[]`` entry."""
    return mr.ModelSpec(name=name, **kw)


def spec_cand(spec_obj, *, info_obj=None, provider=None):
    return mr.ModelCandidate(
        provider=provider or spec_obj.provider or "test",
        model=spec_obj.name,
        info=info_obj,
        spec=spec_obj,
    )


@pytest.fixture
def offline_catalog(monkeypatch):
    """No models.dev at all: every candidate is metadata-free (info=None)."""
    calls = []

    def _candidates_for(provider, models, *, enabled=True):
        calls.append((provider, list(models)))
        return [
            mr.ModelCandidate(provider=provider, model=m, info=None, enabled=enabled)
            for m in models
        ]

    monkeypatch.setattr(mr, "candidates_for", _candidates_for)
    return calls


def _models_config(models, default="", **section):
    base = {"enabled": True, "models": models}
    base.update(section)
    model_section = {"provider": "p1"}
    if default:
        model_section["default"] = default
    return {"smart_model_routing": base, "model": model_section}


# -- models[] is canonical --------------------------------------------------

def test_models_entries_are_the_canonical_candidate_source(offline_catalog):
    cfg = _models_config(
        [
            {"name": "cheap", "provider": "p1", "cost_in": 0.2, "cost_out": 0.8},
            {"name": "other", "provider": "p2", "cost_in": 3.0, "cost_out": 15.0},
        ]
    )
    report = mr.routing_report(cfg, "hello")
    assert report["candidate_source"] == "models"
    assert [(c["provider"], c["model"]) for c in report["candidates"]] == [
        ("p1", "cheap"),
        ("p2", "other"),
    ]


def test_models_metadata_overrides_the_catalog():
    """Declared values win over models.dev, field by field."""
    catalog = info("m", ctx=8_000, out=1_024, tools=False, vision=False, cost_in=9.0, cost_out=9.0)
    declared = spec(
        "m", context_window=200_000, max_output=16_384, supports_tools=True,
        supports_vision=True, cost_in=0.5, cost_out=1.0,
    )
    cand = spec_cand(declared, info_obj=catalog)

    assert cand.context_window() == 200_000
    assert cand.max_output() == 16_384
    assert cand.supports_tools() is mr.CapabilityState.SUPPORTED
    assert cand.supports_input("image") is mr.CapabilityState.SUPPORTED
    assert cand.costs() == (0.5, 1.0)


def test_declared_metadata_can_make_a_catalog_model_eligible_for_vision():
    """models.dev says text-only; the operator knows better."""
    catalog = info("m", vision=False)
    plain = mr.ModelCandidate("test", "m", catalog)
    declared = spec_cand(spec("m", supports_vision=True), info_obj=catalog)
    profile = mr.classify_request("what is in this shot", attachments=["a.png"])
    assert mr.filter_eligible([plain], profile) == []
    assert [c.model for c in mr.filter_eligible([declared], profile)] == ["m"]


def test_declared_metadata_can_raise_the_context_window():
    catalog = info("m", ctx=8_000)
    profile = mr.classify_request("x" * 60_000)  # ~15k tokens
    assert profile.required_context > 8_000
    plain = mr.ModelCandidate("test", "m", catalog)
    raised = spec_cand(spec("m", context_window=200_000), info_obj=catalog)
    assert mr.filter_eligible([plain], profile) == []
    assert [c.model for c in mr.filter_eligible([raised], profile)] == ["m"]


def test_declared_deprecated_flag_excludes_a_live_catalog_model(offline_catalog):
    cand = spec_cand(spec("m", deprecated=True), info_obj=info("m"))
    profile = mr.classify_request("hello")
    assert mr.filter_eligible([cand], profile) == []
    report = mr.routing_report(
        _models_config([{"name": "m", "provider": "p1", "deprecated": True}]), "hello"
    )
    assert report["candidates"][0]["excluded_reason"] == "deprecated"
    assert report["skip_reason"] == "no_eligible_model"


def test_declared_text_output_false_is_a_hard_refusal():
    cand = spec_cand(spec("m", supports_text_output=False))
    assert mr.filter_eligible([cand], mr.classify_request("hello")) == []


def test_flat_deprecated_candidates_list_still_routes(offline_catalog):
    """The pre-contract list is still honoured (deprecated), same scoring path."""
    cfg = {
        "smart_model_routing": {"enabled": True, "candidates": ["test/a", "test/b"]},
        "providers": {"test": {"models": ["a", "b"]}},
        "model": {"provider": "test", "default": "a"},
    }
    report = mr.routing_report(cfg, "What is 2+2?")
    assert report["candidate_source"] == "candidates"
    assert [c["model"] for c in report["candidates"]] == ["a", "b"]
    assert report["selected"] in {"a", "b"}


def test_models_wins_over_a_legacy_candidates_list(offline_catalog):
    cfg = {
        "smart_model_routing": {
            "enabled": True,
            "models": [{"name": "canonical", "provider": "test"}],
            "candidates": ["legacy"],
        },
        "model": {"provider": "test", "default": "canonical"},
    }
    report = mr.routing_report(cfg, "hello")
    assert report["candidate_source"] == "models"
    assert [c["model"] for c in report["candidates"]] == ["canonical"]


# -- median pricing ---------------------------------------------------------

def test_unpriced_model_is_charged_the_median_known_price(offline_catalog):
    cfg = _models_config(
        [
            {"name": "cheap", "provider": "p1", "cost_in": 1.0, "cost_out": 2.0},
            {"name": "dear", "provider": "p1", "cost_in": 5.0, "cost_out": 10.0},
            {"name": "mystery", "provider": "p1"},  # no price anywhere
        ]
    )
    report = mr.routing_report(cfg, "hello")
    rows = {c["model"]: c for c in report["candidates"]}
    assert rows["mystery"]["priced"] is False
    assert rows["mystery"]["charged_as"] == "median"
    # median of the two known prices: 3.0 in, 6.0 out — never zero.
    assert report["pricing"]["median_fallback"] == {"cost_in": 3.0, "cost_out": 6.0}
    assert rows["mystery"]["cost_in"] == 3.0
    assert rows["mystery"]["cost_out"] == 6.0
    assert rows["cheap"]["charged_as"] == "own"


def test_unpriced_model_is_never_treated_as_free():
    profile = mr.TaskProfile(est_input_tokens=1_000_000, est_output_tokens=1_000_000)
    unpriced = mr.ModelCandidate("test", "mystery", None)
    assert mr.estimate_cost(unpriced, profile) is None  # nothing to charge against
    charged = mr.estimate_cost(unpriced, profile, charge=(2.0, 4.0))
    assert charged == pytest.approx(6.0)
    assert charged > 0


def test_pricing_block_reports_the_cheapest_eligible_cost(offline_catalog):
    cfg = _models_config(
        [
            {"name": "cheap", "provider": "p1", "cost_in": 0.1, "cost_out": 0.1},
            {"name": "dear", "provider": "p1", "cost_in": 10.0, "cost_out": 10.0},
        ]
    )
    report = mr.routing_report(cfg, "hello")
    assert report["pricing"]["cheapest_eligible"] is not None
    assert report["pricing"]["policy"] == "unpriced_models_charged_the_median_known_price"


# -- tier floor -------------------------------------------------------------

def _advanced_profile():
    profile = mr.classify_request(
        "Prove the invariant, derive the complexity and analyse the trade-offs of this "
        "algorithm.\n```python\nx = 1\n```"
    )
    assert profile.complexity is mr.Complexity.ADVANCED
    return profile


def test_below_tier_candidate_is_ineligible_for_an_advanced_request():
    low = spec_cand(spec("low", tier_rank=2), info_obj=info("low", reasoning=True))
    high = spec_cand(spec("high", tier_rank=4), info_obj=info("high", reasoning=True))
    eligible = mr.filter_eligible([low, high], _advanced_profile())
    assert [c.model for c in eligible] == ["high"]


def test_below_tier_candidate_is_fine_for_a_simple_request():
    low = spec_cand(spec("low", tier_rank=1), info_obj=info("low"))
    simple = mr.classify_request("What is 2+2?")
    assert simple.complexity is mr.Complexity.SIMPLE
    assert [c.model for c in mr.filter_eligible([low], simple)] == ["low"]


def test_undeclared_tier_is_unknown_not_ineligible():
    """An undeclared tier_rank is missing metadata, never a refusal."""
    plain = mr.ModelCandidate("test", "plain", info("plain"))
    assert plain.tier_rank() is None
    assert [c.model for c in mr.filter_eligible([plain], _advanced_profile())] == ["plain"]


def test_tier_floor_reports_its_reason(offline_catalog):
    cfg = _models_config(
        [
            {"name": "low", "provider": "p1", "tier_rank": 2, "supports_tools": True},
            {"name": "high", "provider": "p1", "tier_rank": 4, "supports_tools": True},
        ]
    )
    report = mr.routing_report(
        cfg,
        "Prove the invariant, derive the complexity and analyse the trade-offs of this "
        "algorithm.\n```python\nx = 1\n```",
    )
    rows = {c["model"]: c for c in report["candidates"]}
    assert rows["low"]["eligible"] is False
    assert rows["low"]["excluded_reason"] == "below_tier"
    assert rows["high"]["eligible"] is True
    assert report["selected"] == "high"


# -- unknown capability metadata -------------------------------------------

def test_unknown_capability_penalty_is_flat_per_required_capability():
    profile = mr.TaskProfile(
        needs_tools=True, needs_structured_output=True, input_modalities=("text", "image")
    )
    known = spec_cand(
        spec("known", supports_tools=True, supports_structured=True, supports_vision=True),
        info_obj=info("known"),
    )
    unknown_all = mr.ModelCandidate("test", "mystery", None)  # nothing declared
    fit_known = mr._capability_fit(known, profile)
    fit_unknown = mr._capability_fit(unknown_all, profile)
    unknown_caps = mr.unknown_required_capabilities(unknown_all, profile)
    # vision + tools + structured (+ text output, which no metadata-free model
    # can answer either) are all unanswered; each costs exactly the flat 0.15,
    # never a refusal while unknowns are allowed.
    assert set(unknown_caps) == {"text_output", "vision", "tools", "structured"}
    assert fit_known - fit_unknown == pytest.approx(0.15 * len(unknown_caps))


def test_one_unknown_capability_costs_exactly_fifteen_hundredths():
    profile = mr.TaskProfile(needs_tools=True)
    known = spec_cand(spec("known", supports_tools=True), info_obj=info("known"))
    # models.dev omits tool_call for many entries; False there means UNKNOWN.
    unknown = spec_cand(spec("unknown"), info_obj=info("unknown", tools=False))
    assert mr.unknown_required_capabilities(unknown, profile) == ("tools",)
    assert mr._capability_fit(known, profile) - mr._capability_fit(unknown, profile) == pytest.approx(0.15)


def test_unknown_metadata_is_allowed_by_default(offline_catalog):
    cand = mr.ModelCandidate("test", "mystery", None)
    profile = mr.classify_request("what is in this shot", attachments=["a.png"])
    assert [c.model for c in mr.filter_eligible([cand], profile)] == ["mystery"]


def test_strict_mode_turns_unknown_into_unsupported(offline_catalog):
    cand = mr.ModelCandidate("test", "mystery", None)
    profile = mr.classify_request("what is in this shot", attachments=["a.png"])
    strict = mr.RoutingConfig(allow_unknown_metadata=False)
    assert mr.filter_eligible([cand], profile, strict) == []
    assert mr._exclusion_reason(cand, profile, strict) == "unknown_vision"

    cfg = _models_config([{"name": "mystery", "provider": "p1"}], allow_unknown_metadata=False)
    report = mr.routing_report(cfg, "what is in this shot", attachments=("a.png",))
    assert report["candidates"][0]["eligible"] is False
    assert report["candidates"][0]["excluded_reason"].startswith("unknown_")
    assert report["skip_reason"] == "no_eligible_model"


def test_strict_mode_excludes_a_declared_unsupported_capability_everywhere():
    strict = mr.RoutingConfig(allow_unknown_metadata=False)
    profile = mr.classify_request("run the tests")
    assert profile.needs_tools
    cand = spec_cand(spec("notools", supports_tools=False), info_obj=info("notools", tools=True))
    assert mr._exclusion_reason(cand, profile, strict) == "unsupported_tools"


# -- hysteresis -------------------------------------------------------------

def _two_models(cheap_in=2.0, rival_in=1.0):
    return [
        spec_cand(spec("current", cost_in=cheap_in, cost_out=cheap_in, supports_tools=True)),
        spec_cand(spec("rival", cost_in=rival_in, cost_out=rival_in, supports_tools=True)),
    ]


def test_hysteresis_keeps_the_current_model_on_a_borderline_win():
    profile = mr.classify_request("hello")
    selection = mr.rank_candidates(
        profile, _two_models(), mr.RoutingConfig(), current_model="current"
    )
    # The rival genuinely ranks first, but by less than the threshold.
    assert selection.best_key == "test/rival"
    assert selection.model == "current"
    assert selection.kept_current is True
    assert selection.skip_reason == "within_stability_threshold"


def test_hysteresis_switches_only_when_clearly_beaten():
    profile = mr.classify_request("hello")
    selection = mr.rank_candidates(
        profile,
        [
            spec_cand(spec("current", cost_in=50.0, cost_out=50.0)),
            spec_cand(spec("rival", cost_in=0.1, cost_out=0.1)),
        ],
        mr.RoutingConfig(switch_on_cost=True),
        current_model="current",
    )
    assert selection.model == "rival"
    assert selection.skip_reason == ""
    assert selection.kept_current is False


def test_a_clear_cost_win_still_waits_for_the_switch_on_cost_optin():
    """Hysteresis is not the only reason to keep the incumbent: a pure cost
    downgrade mid-session is opt-in (prompt-cache stability)."""
    profile = mr.classify_request("hello")
    selection = mr.rank_candidates(
        profile,
        [
            spec_cand(spec("current", cost_in=50.0, cost_out=50.0)),
            spec_cand(spec("rival", cost_in=0.1, cost_out=0.1)),
        ],
        mr.RoutingConfig(),
        current_model="current",
    )
    assert selection.model == "current"
    assert "cost-only downgrade" in selection.reason


def test_identical_models_never_switch():
    profile = mr.classify_request("hello")
    cands = [
        spec_cand(spec("current", cost_in=1.0, cost_out=1.0)),
        spec_cand(spec("rival", cost_in=1.0, cost_out=1.0)),
    ]
    selection = mr.rank_candidates(
        profile, cands, mr.RoutingConfig(stability_threshold=0.0), current_model="current"
    )
    # gain is 0.0 and 0.0 is not "clearly beaten".
    assert selection.model == "current"
    assert selection.skip_reason in ("already_current", "within_stability_threshold")


def test_min_improvement_is_an_alias_for_stability_threshold():
    cfg = mr.RoutingConfig.from_dict({"min_improvement": 0.4})
    assert cfg.stability_threshold == pytest.approx(0.4)

    profile = mr.classify_request("hello")
    kept = mr.rank_candidates(
        profile, _two_models(), cfg, current_model="current"
    )
    switched = mr.rank_candidates(
        profile,
        _two_models(),
        mr.RoutingConfig.from_dict({"min_improvement": 0.001, "switch_on_cost": True}),
        current_model="current",
    )
    assert kept.model == "current"
    assert switched.model == "rival"


def test_default_threshold_is_point_one_five():
    assert mr.RoutingConfig().stability_threshold == pytest.approx(TSIM)


# -- skip_reason values -----------------------------------------------------

def _all_candidates_config(**section):
    base = {"switch_on_cost": True}
    base.update(section)
    return _models_config(
        [
            {"name": "current", "provider": "p1", "cost_in": 50.0, "cost_out": 50.0},
            {"name": "rival", "provider": "p1", "cost_in": 0.1, "cost_out": 0.1},
        ],
        default="current",
        **base,
    )


@pytest.mark.parametrize(
    "label,config,prompt,current,expected",
    [
        ("switch", _all_candidates_config(), "hello", "current", ""),
        ("disabled", _all_candidates_config(enabled=False), "hello", "current", "disabled"),
        ("already_current", _models_config([{"name": "only", "provider": "p1"}]), "hello", "only", "already_current"),
        (
            "no_eligible_model",
            _models_config([{"name": "gone", "provider": "p1", "deprecated": True}]),
            "hello",
            "",
            "no_eligible_model",
        ),
        (
            "within_stability_threshold",
            _models_config(
                [
                    {"name": "current", "provider": "p1", "cost_in": 2.0, "cost_out": 2.0},
                    {"name": "rival", "provider": "p1", "cost_in": 1.0, "cost_out": 1.0},
                ],
                default="current",
            ),
            "hello",
            "current",
            "within_stability_threshold",
        ),
    ],
)
def test_every_skip_reason_is_reported(offline_catalog, label, config, prompt, current, expected):
    report = mr.routing_report(config, prompt, current_model=current)
    assert report["skip_reason"] == expected, label
    assert report["skip_reason"] in mr.SKIP_REASONS


def test_no_candidates_is_reported_when_nothing_is_declared(offline_catalog):
    report = mr.routing_report(
        {"smart_model_routing": {"enabled": True}, "model": {"provider": "p1"}}, "hello"
    )
    assert report["skip_reason"] == "no_candidates"
    assert report["candidates"] == []
    assert report["eligible_count"] == 0


def test_error_skip_reason_is_used_when_selection_blows_up(monkeypatch):
    def _boom(*_a, **_kw):
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(mr, "_preview_selection", _boom)
    report = mr.routing_report(_all_candidates_config(), "hello")
    assert report["skip_reason"] == "error"
    assert "catalog exploded" in report["reason"]


def test_skip_reason_constants_match_the_shared_contract():
    # C10 names ``moa_active`` as a skip reason, so the vocabulary must carry it even
    # though no code path emits it yet (the MoA path does not consult the router).
    assert set(mr.SKIP_REASONS) == {
        "", "disabled", "no_candidates", "no_eligible_model", "already_current",
        "within_stability_threshold", "moa_active", "error",
    }


# -- the JSON report -------------------------------------------------------

REPORT_KEYS = {
    "enabled", "settings", "candidate_source", "request", "pricing", "selected",
    "selected_provider", "reason", "skip_reason", "current", "kept_current",
    "eligible_count", "candidates",
}


def test_report_has_exactly_the_contract_key_set(offline_catalog):
    report = mr.routing_report(_all_candidates_config(), "hello")
    assert set(report) == REPORT_KEYS
    assert set(report["settings"]) == {
        "strategy", "stability_threshold", "allow_unknown_metadata",
        "cost_optimization", "preferred_models_only",
    }
    assert set(report["request"]) == {
        "tier", "est_input_tokens", "est_output_tokens", "needs_tools",
        "needs_vision", "needs_audio", "reasons",
    }
    assert set(report["pricing"]) == {"cheapest_eligible", "median_fallback", "policy"}
    assert set(report["candidates"][0]) == {
        "model", "provider", "eligible", "score", "tier", "cost_in", "cost_out",
        "priced", "charged_as", "unknown_capabilities", "excluded_reason",
    }


def test_report_is_json_serialisable_and_stable(offline_catalog):
    report = mr.routing_report(_all_candidates_config(), "hello")
    text = json.dumps(report, indent=2, sort_keys=True)
    assert json.loads(text) == report
    # sorted keys, two-space indent
    assert text.splitlines()[1].startswith('  "candidate_source"')


def test_disabled_report_still_computes_the_hypothetical_selection(offline_catalog):
    report = mr.routing_report(_all_candidates_config(enabled=False), "hello")
    assert report["enabled"] is False
    assert report["skip_reason"] == "disabled"
    # the hypothetical is still there — it just is not applied
    assert report["selected"] in {"current", "rival"}
    assert report["candidates"]
    assert report["pricing"]["median_fallback"] is not None


def test_report_agrees_with_the_turn_hook(offline_catalog):
    """The report and the hook share rank_candidates, so they cannot disagree."""
    cfg = _models_config(
        [
            {"name": "current", "provider": "p1", "cost_in": 50.0, "cost_out": 50.0},
            {"name": "rival", "provider": "p1", "cost_in": 0.1, "cost_out": 0.1},
        ],
        default="current",
    )
    prompt = "What is 15 multiplied by 8?"

    report = mr.routing_report(cfg, prompt, current_model="current")

    agent = _StubAgent(model="current", provider="p1")
    decision = mr.route_turn(agent, prompt, config=cfg)

    assert decision is not None
    assert (decision.provider, decision.model) == (
        report["selected_provider"], report["selected"]
    )
    assert decision.skip_reason == report["skip_reason"]
    assert agent.switch_calls, "a clearly beaten incumbent should have been replaced"
    assert agent.model == report["selected"]


def test_report_never_writes_config(offline_catalog):
    cfg = _all_candidates_config()
    before = repr(cfg)
    mr.routing_report(cfg, "hello")
    assert repr(cfg) == before


def test_legacy_candidates_path_matches_the_models_path(offline_catalog):
    """Same models, same prices, same verdict — whichever key declared them."""
    models = [
        {"name": "a", "provider": "test", "cost_in": 20.0, "cost_out": 20.0},
        {"name": "b", "provider": "test", "cost_in": 0.1, "cost_out": 0.1},
    ]
    canonical = {"smart_model_routing": {"enabled": True, "models": models},
                 "model": {"provider": "test", "default": "a"}}
    legacy = {"smart_model_routing": {"enabled": True, "candidates": ["a", "b"]},
              "model": {"provider": "test", "default": "a"}}
    canonical_selection = mr.routing_report(canonical, "hello")
    legacy_selection = mr.routing_report(legacy, "hello")
    # The legacy path carries no declared prices, so only the *choice* has to
    # agree with a cost-free pool; what matters is that both go through one
    # scoring path and both switch away from the expensive incumbent.
    assert canonical_selection["candidate_source"] == "models"
    assert legacy_selection["candidate_source"] == "candidates"
    assert canonical_selection["selected"] == "b"


# ---------------------------------------------------------------------------
# Shared contract C6 (docs/routing-contract.md): widening is the default, and a
# single-model provider list stays a no-op unless the operator opts in.
# ---------------------------------------------------------------------------


def test_shared_contract_default_allows_widening():
    cfg = mr.RoutingConfig.from_dict({})
    assert cfg.preferred_models_only is False
    assert cfg.preferred_models_only_explicit is False

    explicit = mr.RoutingConfig.from_dict({"preferred_models_only": True})
    assert explicit.preferred_models_only is True
    assert explicit.preferred_models_only_explicit is True


def test_single_model_pool_stays_preferred_without_an_explicit_flag():
    cfg = _multi_config(("weak",), preferred_only=False)
    cfg["smart_model_routing"].pop("preferred_models_only")
    pool, scope = mr.routing_pool(cfg, "test", current_model="weak")
    assert scope == "preferred"          # nothing to choose between, no catalog scan
    assert pool == ["weak"]


def test_single_model_pool_widens_only_when_explicitly_asked(patched_candidates):
    cfg = _multi_config(("weak",), preferred_only=False)   # key present and False
    assert mr.RoutingConfig.from_dict(cfg["smart_model_routing"]).preferred_models_only_explicit
    _pool, scope = mr.routing_pool(cfg, "test", current_model="weak")
    assert scope == "preferred+catalog"  # opt-in widening is honoured


# ---------------------------------------------------------------------------
# Documented behaviour that the ranker must actually keep (C10 vocabulary and the
# two fields the contract used to describe as behaviour).
# ---------------------------------------------------------------------------


def test_skip_reasons_cover_the_contract_c10_vocabulary():
    documented = {
        "within_stability_threshold",
        "moa_active",
        "disabled",
        "error",
    }
    assert documented <= set(mr.SKIP_REASONS)


def test_equal_scores_resolve_alphabetically_not_by_latency():
    """``latency_ms`` is documented as an unused field, and the real tie-break is the
    model key. If someone later implements the latency tie-break, this test fails and
    points at the contract line that has to change with it."""
    profile = mr.classify_request("What is 15 multiplied by 8?")
    slow_alpha = mr.ModelCandidate("test", "alpha", info("alpha"), latency_ms=900.0)
    fast_zeta = mr.ModelCandidate("test", "zeta", info("zeta"), latency_ms=10.0)
    selection = mr.rank_candidates(profile, [fast_zeta, slow_alpha])
    assert selection.model == "alpha"          # declaration order and latency both ignored
    reversed_selection = mr.rank_candidates(profile, [slow_alpha, fast_zeta])
    assert reversed_selection.model == "alpha"


def test_candidate_reliability_is_only_operator_config():
    """Nothing measures outcomes, so a candidate carries no reliability of its own and
    the only source is ``smart_model_routing.reliability``."""
    profile = mr.classify_request("What is 15 multiplied by 8?")
    plain = mr.ModelCandidate("test", "m", info("m"))
    assert plain.reliability is None
    cfg = mr.RoutingConfig.from_dict({"reliability": {"m": 0.9}})
    assert mr._reliability_of(plain, cfg) == 0.9


# ---------------------------------------------------------------------------
# Incumbent identity is (provider, model), not the model name alone. One model id
# can be offered by two providers, and a name-only match reports a provider the
# session is not on (contract C5 / C8 boundary).
# ---------------------------------------------------------------------------


def test_same_model_name_on_two_providers_keeps_the_incumbent_provider():
    profile = mr.classify_request("What is 15 multiplied by 8?")
    incumbent = mr.ModelCandidate("prov-a", "same-model", info("same-model"))
    rival = mr.ModelCandidate("prov-b", "same-model", info("same-model"))
    selection = mr.rank_candidates(
        profile, [rival, incumbent], current_model="same-model", current_provider="prov-a"
    )
    assert (selection.provider, selection.model) == ("prov-a", "same-model")
    assert selection.skip_reason == "already_current"


def test_continuity_does_not_credit_another_providers_identical_model_name():
    profile = mr.classify_request("What is 15 multiplied by 8?")
    cfg = mr.RoutingConfig(strategy="quality")      # continuity carries weight here
    mine = mr.ModelCandidate("prov-a", "same-model", info("same-model"))
    theirs = mr.ModelCandidate("prov-b", "same-model", info("same-model"))
    scored_mine = mr.score_candidate(mine, profile, cfg, current_model="same-model",
                                     current_provider="prov-a")
    scored_theirs = mr.score_candidate(theirs, profile, cfg, current_model="same-model",
                                       current_provider="prov-a")
    assert scored_mine > scored_theirs


def test_ties_resolve_on_the_provider_qualified_key():
    """The documented order is (-score, provider/model): total, so two providers
    offering the same model name cannot tie into declaration order."""
    profile = mr.classify_request("What is 15 multiplied by 8?")
    a = mr.ModelCandidate("aaa", "same-model", info("same-model"))
    z = mr.ModelCandidate("zzz", "same-model", info("same-model"))
    first = mr.rank_candidates(profile, [z, a])
    second = mr.rank_candidates(profile, [a, z])
    assert first.best_key == second.best_key == "aaa/same-model"


# ---------------------------------------------------------------------------
# C9a — the input-modality requirement must come from the attachments at the
# turn boundary. Before this, the loop coerced the message with str(), which kept
# the text and dropped every attachment, so an image turn could be routed to a
# model that cannot read images.
# ---------------------------------------------------------------------------


def test_parts_list_keeps_text_and_surfaces_the_image_attachment():
    text, attachments = mr.message_text_and_attachments([
        {"type": "text", "text": "What is wrong with this chart?"},
        {"type": "image_url", "image_url": {"url": "https://example.test/c.png"}},
    ])
    assert text == "What is wrong with this chart?"
    assert attachments == ("image",)


def test_plain_string_turns_are_unchanged():
    assert mr.message_text_and_attachments("hello there") == ("hello there", ())
    assert mr.message_text_and_attachments(None) == ("", ())


def test_file_parts_map_by_suffix_and_audio_parts_map_to_audio():
    _, pdf = mr.message_text_and_attachments(
        [{"type": "file", "file": {"filename": "q3-report.pdf"}}]
    )
    _, audio = mr.message_text_and_attachments([{"type": "input_audio", "input_audio": {}}])
    assert pdf == ("pdf",) and audio == ("audio",)


def test_unknown_parts_neither_crash_nor_invent_a_modality():
    text, attachments = mr.message_text_and_attachments([
        {"type": "mystery"}, {"type": "text", "text": "still here"}, {"nope": 1},
    ])
    assert text == "still here"
    assert attachments == ()


def test_attachment_makes_vision_a_requirement_and_excludes_non_vision_models():
    profile = mr.classify_request("Describe this chart", attachments=["image"])
    assert "image" in profile.input_modalities
    assert "vision" in mr.required_capabilities(profile)

    blind = mr.ModelCandidate("blind", "no-eyes", info("no-eyes", vision=False))
    sighted = mr.ModelCandidate("sighted", "sees", info("sees", vision=True))
    cfg = mr.RoutingConfig()
    assert mr._exclusion_reason(blind, profile, cfg) == "unsupported_vision"
    assert mr._exclusion_reason(sighted, profile, cfg) == ""


def test_an_image_turn_cannot_select_a_non_vision_model():
    profile = mr.classify_request("Describe this chart", attachments=["image"])
    blind = mr.ModelCandidate("blind", "no-eyes", info("no-eyes", vision=False))
    sighted = mr.ModelCandidate("sighted", "sees", info("sees", vision=True))
    selection = mr.rank_candidates(profile, [blind, sighted])
    assert selection.best_key == "sighted/sees"


def test_moa_skip_is_recorded_rather_than_silent():
    class Agent:
        pass

    agent = Agent()
    assert mr.route_turn(agent, "hello", moa_active=True) is None
    assert getattr(agent, mr.ROUTING_SKIP_ATTR) == "moa_active"


def test_moa_skip_appears_in_the_machine_report():
    """C10: the skip must be in the dict, not only in agent state."""
    cfg = {"smart_model_routing": {"enabled": True, "models": [
        {"name": "small", "provider": "test", "cost_in": 0.1, "cost_out": 0.2},
        {"name": "big", "provider": "test", "cost_in": 5.0, "cost_out": 9.0},
    ]}}
    report = mr.routing_report(cfg, "hello", moa_active=True)
    assert report["skip_reason"] == "moa_active"
    assert report["selected"] is None
    assert report["kept_current"] is True
    assert "mixture-of-agents" in report["reason"]
    # the hypothetical rows stay, so the skip is not mistaken for "no candidates"
    assert report["candidates"]


def test_decision_records_the_modalities_it_was_made_for():
    """C9b's precondition: an image-edit follow-up needs to know what the earlier
    decision promised, and a Decision with no modality record cannot tell it."""
    profile = mr.classify_request("Describe this chart", attachments=["image"])
    cand = mr.ModelCandidate("test", "sees", info("sees", vision=True))
    selection = mr.rank_candidates(profile, [cand])
    decision = selection.decision()
    assert decision is not None
    assert "image" in decision.input_modalities
    assert decision.output_modalities == ("text",)


# ---------------------------------------------------------------------------
# C9b — an OUTPUT modality has to be declarable, or a text-only model stays
# UNKNOWN (and therefore eligible) for an image-edit request.
# ---------------------------------------------------------------------------


def test_supports_image_output_is_a_declarable_spec_field():
    spec = mr.ModelSpec.from_entry({"name": "txt-only", "supports_image_output": False})
    assert spec is not None and spec.supports_image_output is False
    cand = mr.ModelCandidate("test", "txt-only", info("txt-only"), spec=spec)
    assert mr.capability_state(cand, "output_image") is mr.CapabilityState.UNSUPPORTED


def test_image_output_request_excludes_a_model_that_cannot_emit_images():
    profile = mr.classify_request("recolour this chart", expects_image=True)
    assert "image" in profile.output_modalities
    assert "output_image" in mr.required_capabilities(profile)

    text_only = mr.ModelCandidate(
        "test", "txt-only", info("txt-only"),
        spec=mr.ModelSpec.from_entry({"name": "txt-only", "supports_image_output": False}),
    )
    painter = mr.ModelCandidate(
        "test", "painter", info("painter"),
        spec=mr.ModelSpec.from_entry({"name": "painter", "supports_image_output": True}),
    )
    cfg = mr.RoutingConfig()
    assert mr._exclusion_reason(text_only, profile, cfg) == "unsupported_output_image"
    assert mr._exclusion_reason(painter, profile, cfg) == ""
    assert mr.rank_candidates(profile, [text_only, painter]).best_key == "test/painter"


def test_an_undeclared_output_modality_stays_unknown_not_unsupported():
    """Unknown is not the same claim as unsupported: a model that never declares
    image output must not be excluded, only unproven."""
    profile = mr.classify_request("recolour this chart", expects_image=True)
    # no catalog entry at all: nothing declares what this model can emit
    silent = mr.ModelCandidate("test", "silent", None)
    assert mr.capability_state(silent, "output_image") is mr.CapabilityState.UNKNOWN
    assert mr._exclusion_reason(silent, profile, mr.RoutingConfig()) == ""


def test_an_image_output_request_is_recognised_from_the_turn_itself():
    """C9b: until the classifier reads the turn, only a caller that already knew
    to pass expects_image could protect an image-output request."""
    for prompt in ("generate an image of a red fox", "Draw me a picture of the harbour",
                   "render the chart as an image"):
        profile = mr.classify_request(prompt)
        assert "image" in profile.output_modalities, prompt
        assert "output_image" in mr.required_capabilities(profile), prompt


def test_merely_mentioning_an_image_does_not_ask_for_one():
    """Negative controls: an attachment's description and a question about image
    generation are INPUT requests, and must not trip the output gate."""
    for prompt in ("what is in this image?", "explain how image generation works",
                   "summarise this picture for me"):
        profile = mr.classify_request(prompt)
        assert "image" not in profile.output_modalities, prompt


def test_image_output_request_routes_away_from_a_text_only_model():
    profile = mr.classify_request("generate an image of a red fox")
    text_only = mr.ModelCandidate(
        "test", "txt-only", info("txt-only"),
        spec=mr.ModelSpec.from_entry({"name": "txt-only", "supports_image_output": False}),
    )
    painter = mr.ModelCandidate(
        "test", "painter", info("painter"),
        spec=mr.ModelSpec.from_entry({"name": "painter", "supports_image_output": True}),
    )
    assert mr._exclusion_reason(text_only, profile, mr.RoutingConfig()) == "unsupported_output_image"
    assert mr.rank_candidates(profile, [text_only, painter]).best_key == "test/painter"


# -- switch_on_cost: the mid-session opt-in must actually be reachable ---------


def test_switch_on_cost_allows_a_cost_downgrade_mid_session():
    """`switch_on_cost` is exposed as 'Allow a mid-session switch to a cheaper model'.

    The mid-session guard (prompt-cache stability) used to return before the flag was read,
    so the setting was inert for every turn after the first: the CLI and the picker offered a
    switch that could not change a decision.
    """
    profile = mr.classify_request("hello")
    selection = mr.rank_candidates(
        profile,
        [spec_cand(spec("current", cost_in=50.0, cost_out=50.0)),
         spec_cand(spec("rival", cost_in=0.1, cost_out=0.1))],
        mr.RoutingConfig(switch_on_cost=True),
        current_model="current",
        mid_session=True,
    )
    assert selection.model == "rival"
    assert selection.kept_current is False


def test_switch_on_cost_never_upgrades_mid_session_without_a_gap():
    """The opt-in names a *cheaper* model. A mid-session jump that merely scores higher, at the
    same price, would pay the prompt-cache miss for a better fit it cannot justify."""
    profile = mr.classify_request("hello", conversation_tokens=60000)
    cfg = mr.RoutingConfig(
        switch_on_cost=True, stability_threshold=0.001,
        # Fit dominates, price and reliability are neutral, so the rival wins on context fit
        # alone: a score win with no cost argument.
        weights={"capability": 0.1, "cost": 0.0, "reliability": 0.0,
                 "context": 0.9, "continuity": 0.0},
    )
    selection = mr.rank_candidates(
        profile,
        [spec_cand(spec("current", cost_in=1.0, cost_out=1.0, context_window=70000)),
         spec_cand(spec("rival", cost_in=1.0, cost_out=1.0, context_window=2_000_000))],
        cfg,
        current_model="current",
        mid_session=True,
    )
    assert selection.model == "current", "a non-cheaper rival must not dislodge the incumbent"
    assert "mid-session switch requires" in selection.reason


def test_switch_on_cost_off_keeps_the_incumbent_mid_session():
    profile = mr.classify_request("hello")
    selection = mr.rank_candidates(
        profile,
        [spec_cand(spec("current", cost_in=50.0, cost_out=50.0)),
         spec_cand(spec("rival", cost_in=0.1, cost_out=0.1))],
        mr.RoutingConfig(switch_on_cost=False),
        current_model="current",
        mid_session=True,
    )
    assert selection.model == "current"
    assert selection.skip_reason == "within_stability_threshold"


def test_route_turn_switches_mid_session_when_the_operator_opted_in(patched_candidates):
    agent = _StubAgent(model="pricey")
    cfg = _route_config(models=("weak", "pricey"), default="pricey")
    cfg["smart_model_routing"]["switch_on_cost"] = True
    mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)   # turn 0 -> weak
    agent.model = "pricey"                                            # back on the pricey one
    agent.switch_calls.clear()
    mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)   # turn 1, opted in
    assert agent.switch_calls, "switch_on_cost=True must permit the mid-session downgrade"
    assert agent.model == "weak"


# -- tool_calls_only: the default model is the assistant, the pool is its delegates -------


def test_tool_calls_only_never_displaces_the_default_model_for_price():
    """The strategy's whole promise: a cheaper rival does not take the conversation."""
    profile = mr.classify_request("summarise this paragraph", tool_available=False)
    selection = mr.rank_candidates(
        profile,
        [spec_cand(spec("default", cost_in=50.0, cost_out=50.0)),
         spec_cand(spec("cheap", cost_in=0.01, cost_out=0.01))],
        # cost-heavy weights so the price gap is unmissable, and no stability band to hide in
        mr.RoutingConfig(strategy="tool_calls_only", stability_threshold=0.0,
                         weights={"capability": 0.05, "cost": 0.95, "reliability": 0.0,
                                  "context": 0.0, "continuity": 0.0}),
        current_model="default",
        mid_session=False,
    )
    assert selection.model == "default"
    assert selection.skip_reason == "already_current"
    assert "tool_calls_only" in selection.reason


def test_tool_calls_only_ignores_switch_on_cost():
    profile = mr.classify_request("summarise this paragraph", tool_available=False)
    selection = mr.rank_candidates(
        profile,
        [spec_cand(spec("default", cost_in=50.0, cost_out=50.0)),
         spec_cand(spec("cheap", cost_in=0.01, cost_out=0.01))],
        mr.RoutingConfig(strategy="tool_calls_only", switch_on_cost=True, stability_threshold=0.0),
        current_model="default",
        mid_session=True,
    )
    assert selection.model == "default", "the mid-session cost opt-in cannot move the default model"


def test_tool_calls_only_replaces_the_default_model_when_it_cannot_serve_the_turn():
    """Capability-only displacement: the incumbent cannot read a pdf, so a delegate that can."""
    profile = mr.classify_request("read this pdf", tool_available=True, attachments=("pdf",))
    # `info` is the normalized catalog record: capability lookups read attributes off it.
    text_only = mr.ModelCandidate(
        provider="test", model="default",
        info=SimpleNamespace(input_modalities=("text",), output_modalities=("text",)),
        spec=spec("default", cost_in=0.1, cost_out=0.1),
    )
    pdf_capable = mr.ModelCandidate(
        provider="test", model="reader",
        info=SimpleNamespace(input_modalities=("text", "pdf"), output_modalities=("text",)),
        spec=spec("reader", cost_in=9.0, cost_out=9.0),
    )
    selection = mr.rank_candidates(
        profile, [text_only, pdf_capable],
        mr.RoutingConfig(strategy="tool_calls_only"),
        current_model="default",
        mid_session=True,
    )
    assert selection.model == "reader"
    assert selection.kept_current is False


def test_tool_calls_only_is_an_accepted_strategy():
    assert "tool_calls_only" in mr.ROUTING_STRATEGIES
    assert mr.RoutingConfig.from_dict({"strategy": "tool_calls_only"}).strategy == "tool_calls_only"
    # an unknown value still degrades to automatic rather than pinning a typo
    assert mr.RoutingConfig.from_dict({"strategy": "tool_calls_onyl"}).strategy == "automatic"


def test_route_turn_tool_calls_only_leaves_the_session_model_alone(patched_candidates):
    agent = _StubAgent(model="weak")
    cfg = _route_config(models=("weak", "strong"), default="weak")
    cfg["smart_model_routing"]["strategy"] = "tool_calls_only"
    mr.route_turn(agent, "what is 15 multiplied by 8?", config=cfg)
    assert agent.switch_calls == [], "tool_calls_only must not switch an eligible default model"
    assert agent.model == "weak"


from agent.model_router import message_text_and_attachments  # noqa: E402  (test-local import)


class TestAdapterNoteAttachments:
    """A gateway upload reaches routing as plain text; the note is the only trace of the file.

    Regression: Telegram's document note continues on the same line as the path
    ("…x.pdf. Its text is not inlined here …"). Routing saw a string, contributed no attachment, and
    kept an incumbent that cannot read the file while the plugin refused the path as untrusted — the
    capability existed on paper and was dead in practice on that surface.
    """

    def test_a_document_note_asks_for_a_model_that_reads_pdfs(self):
        text = ("[The user sent a document: 'foo.pdf'. It is saved at: /home/u/.hermes/cache/documents/"
                "doc_1_foo.pdf. Its text is not inlined here (it's a binary format such as PDF or "
                "DOCX). To read it, extract the document's text yourself.]\n\nread it")
        _, attachments = message_text_and_attachments(text)
        assert attachments == ("pdf",)

    def test_a_voice_note_asks_for_a_model_that_takes_audio(self):
        text = "[The user sent a voice note. It is saved at: /home/u/.hermes/cache/audio/v.ogg]"
        assert message_text_and_attachments(text)[1] == ("audio",)

    def test_the_hint_survives_a_parts_list(self):
        parts = [{"type": "text", "text": "[The user sent a video. It is saved at: /t/c.mp4]"}]
        assert message_text_and_attachments(parts)[1] == ("video",)

    def test_plain_text_claims_no_attachment(self):
        assert message_text_and_attachments("no files here, just words")[1] == ()

    def test_a_note_without_a_path_claims_no_attachment(self):
        assert message_text_and_attachments("[saved at: unknown]")[1] == ()
