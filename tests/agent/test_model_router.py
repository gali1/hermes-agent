"""Tests for agent.model_router — deterministic model selection.

No network: every candidate carries a hand-built ``ModelInfo``.
"""

import os
import sys

import pytest

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
    assert cfg.stability_threshold == pytest.approx(0.08)


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
