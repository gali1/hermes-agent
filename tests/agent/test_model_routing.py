"""Tests for the deterministic model-routing engine.

The engine is arithmetic, so these tests are exact: no fakes, no clock, no network. The two
properties that matter most are that an ineligible model can never win on price, and that a
disabled config is a strict no-op.
"""

from __future__ import annotations

import pytest

import agent.model_routing as model_routing
from agent.model_routing import (
    TIERS, ModelOption, RequestProfile, classify_request, estimate_cost, estimate_tokens,
    _exclusion_reason, explain_routing, is_eligible, maybe_apply_model_routing,
    options_from_config, parse_settings, score_model, select_model,
)

CHEAP_WEAK = ModelOption(name="cheap", cost_in=0.1, cost_out=0.2, tier_rank=0)
STRONG = ModelOption(name="strong", cost_in=5.0, cost_out=15.0, tier_rank=5)
MID = ModelOption(name="mid", cost_in=1.0, cost_out=3.0, tier_rank=3)
VISION = ModelOption(name="vision", cost_in=2.0, cost_out=6.0, tier_rank=4, supports_vision=True)
NO_TOOLS = ModelOption(name="no-tools", cost_in=0.1, cost_out=0.1, tier_rank=3, supports_tools=False)
TEXT_ONLY = ModelOption(name="text-only", cost_in=0.1, cost_out=0.2, tier_rank=5, supports_vision=False)


class TestTokenEstimate:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_roughly_four_chars_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_clamped_on_huge_input(self):
        assert estimate_tokens("a" * 20_000_000) <= 2_000_000


class TestClassification:
    def test_trivial_is_simple(self):
        assert classify_request("what is 15 * 8?").tier == "simple"

    def test_greeting_is_simple(self):
        assert classify_request("hello").tier == "simple"

    def test_code_raises_tier(self):
        assert TIERS.index(classify_request("```\nx=1\n```").tier) >= 3

    def test_multi_part_is_at_least_moderate(self):
        body = "Please do these:\n1. check the logs\n2. restart the service"
        assert TIERS.index(classify_request(body).tier) >= 2

    def test_reasoning_verbs_raise_tier(self):
        assert TIERS.index(classify_request("analyse the trade-off and prove it").tier) >= 3

    def test_long_input_raises_tier(self):
        assert TIERS.index(classify_request("word " * 3000).tier) >= 3

    def test_modality_is_specialized(self):
        assert classify_request("look at this", has_images=True).tier == "specialized"

    def test_short_but_deep_is_not_downgraded(self):
        """Length must not dominate: a short question can still need reasoning."""
        assert TIERS.index(classify_request("why does the scheduler deadlock?").tier) >= 3

    def test_profile_records_reasons(self):
        assert classify_request("```code```").reasons


class TestEligibility:
    def test_vision_required_excludes_explicit_text_only(self):
        profile = classify_request("describe", has_images=True)
        assert not is_eligible(TEXT_ONLY, profile)
        assert is_eligible(VISION, profile)

    def test_unknown_vision_is_eligible_not_excluded(self):
        """Spec §9.2: a silent catalog is not proof the model lacks the capability."""
        profile = classify_request("describe", has_images=True)
        assert is_eligible(STRONG, profile)          # STRONG says nothing about vision

    def test_unknown_capability_is_penalised(self):
        profile = classify_request("describe", has_images=True)
        known = score_model(VISION, profile, cheapest=0.01)
        unknown = score_model(STRONG, profile, cheapest=0.01)
        assert known > unknown

    def test_tools_required_excludes_no_tool_model(self):
        profile = classify_request("do it", needs_tools=True)
        assert not is_eligible(NO_TOOLS, profile)

    def test_context_window_enforced(self):
        profile = RequestProfile("moderate", est_input_tokens=100_000, est_output_tokens=500)
        assert not is_eligible(ModelOption(name="small", context_window=8_000), profile)
        assert is_eligible(ModelOption(name="big", context_window=200_000), profile)

    def test_zero_means_unknown_not_zero(self):
        """A catalog that omits a window must not be read as 'cannot hold anything'."""
        profile = RequestProfile("moderate", est_input_tokens=50_000, est_output_tokens=500)
        assert is_eligible(ModelOption(name="unknown", context_window=0), profile)


class TestScoring:
    def test_capability_gate_beats_price(self):
        """A cheaper model below the request tier must not outrank a capable one."""
        profile = classify_request("architect and debug this refactor")
        decision = select_model([CHEAP_WEAK, STRONG], profile)
        assert decision.model == "strong"

    def test_cheap_wins_for_trivial_work(self):
        profile = classify_request("hello")
        decision = select_model([CHEAP_WEAK, STRONG], profile)
        assert decision.model == "cheap"

    def test_cost_estimated_from_prices(self):
        profile = RequestProfile("routine", est_input_tokens=1_000, est_output_tokens=1_000)
        assert estimate_cost(ModelOption(name="m", cost_in=1.0, cost_out=2.0), profile) == pytest.approx(0.003)

    def test_score_is_lower_for_expensive_equals(self):
        profile = classify_request("hello")
        cheap = score_model(CHEAP_WEAK, profile, cheapest=0.001)
        pricey = score_model(STRONG, profile, cheapest=0.001)
        assert cheap > pricey


class TestSelection:
    def test_no_eligible_model_returns_none(self):
        profile = classify_request("see image", has_images=True)
        decision = select_model([TEXT_ONLY], profile)
        assert decision.model is None
        assert "no eligible" in decision.reason

    def test_hysteresis_keeps_current_when_close(self):
        profile = classify_request("hello")
        decision = select_model([CHEAP_WEAK, STRONG], profile, current="strong", min_improvement=0.9)
        assert decision.model == "strong"
        assert "keeping current" in decision.reason

    def test_hard_upgrade_still_switches(self):
        """Hysteresis must not pin a model that cannot do the job."""
        profile = classify_request("```refactor this```")
        decision = select_model([CHEAP_WEAK, STRONG], profile, current="cheap", min_improvement=0.1)
        assert decision.model == "strong"

    def test_empty_candidates(self):
        assert select_model([], classify_request("hi")).model is None

    def test_decision_reports_candidate_count(self):
        assert select_model([CHEAP_WEAK, STRONG, MID], classify_request("hi")).candidates == 3


class TestConfigParsing:
    def test_malformed_entries_skipped(self):
        opts = options_from_config({"models": [{"name": "ok"}, {"nope": 1}, "junk", {"name": ""}]})
        assert [o.name for o in opts] == ["ok"]

    def test_non_list_models(self):
        assert options_from_config({"models": "nope"}) == []

    def test_prices_and_flags_parsed(self):
        opts = options_from_config({"models": [{
            "name": "m", "provider": "p", "cost_in": "1.5", "cost_out": 2,
            "supports_vision": True, "tier_rank": 4}]})
        assert opts[0].cost_in == 1.5 and opts[0].supports_vision and opts[0].tier_rank == 4


class _Agent:
    def __init__(self, config=None, model="strong", provider="p"):
        self.config = config or {}
        self.model = model
        self.provider = provider
        self.requested_provider = provider
        self.switched = None


class TestMaybeApply:
    def test_disabled_is_strict_noop(self):
        agent = _Agent({"smart_model_routing": {"enabled": False, "models": [{"name": "cheap"}]}})
        assert maybe_apply_model_routing(agent, "hello") is None
        assert agent.switched is None

    def test_missing_config_is_noop(self):
        assert maybe_apply_model_routing(_Agent({}), "hello") is None

    def test_enabled_but_no_models_is_noop(self):
        agent = _Agent({"smart_model_routing": {"enabled": True, "models": []}})
        assert maybe_apply_model_routing(agent, "hello") is None

    def test_bad_config_shape_is_noop(self):
        agent = _Agent({"smart_model_routing": "yes"})
        assert maybe_apply_model_routing(agent, "hello") is None

    def test_switch_failure_is_swallowed(self, monkeypatch):
        """A routing failure must never break the turn."""
        agent = _Agent({"smart_model_routing": {"enabled": True, "models": [
            {"name": "cheap", "cost_in": 0.1, "tier_rank": 0}]}}, model="strong")

        def _boom(*a, **k):
            raise RuntimeError("switch exploded")

        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", _boom)
        assert maybe_apply_model_routing(agent, "hello") is None


class TestSettingsContract:
    """One authoritative smart_model_routing contract, both historical spellings accepted."""

    def test_defaults(self):
        st = parse_settings({})
        assert st.enabled is False and st.strategy == "automatic"
        assert st.stability_threshold == 0.15 and st.allow_unknown_metadata is True

    def test_lab_spelling(self):
        st = parse_settings({"enabled": True, "stability_threshold": 0.4, "strategy": "cost"})
        assert st.enabled and st.stability_threshold == 0.4 and st.strategy == "cost"

    def test_earlier_spelling_still_works(self):
        """Backward compatibility: min_improvement was the original name for the same knob."""
        assert parse_settings({"min_improvement": 0.3}).stability_threshold == 0.3

    def test_lab_spelling_wins_when_both_present(self):
        st = parse_settings({"stability_threshold": 0.5, "min_improvement": 0.1})
        assert st.stability_threshold == 0.5

    def test_bad_strategy_falls_back(self):
        assert parse_settings({"strategy": "banana"}).strategy == "automatic"

    def test_bad_threshold_falls_back(self):
        assert parse_settings({"stability_threshold": "junk"}).stability_threshold == 0.15

    def test_strategy_changes_cost_weight(self):
        assert parse_settings({"strategy": "quality"}).cost_weight == 0.0
        assert parse_settings({"strategy": "cost"}).cost_weight > parse_settings({}).cost_weight

    def test_strict_mode_excludes_unknown(self):
        profile = classify_request("do it", needs_tools=True)
        unknown_tools = ModelOption(name="u", tier_rank=3)
        assert is_eligible(unknown_tools, profile, True) is True
        assert is_eligible(unknown_tools, profile, False) is False


class TestExplain:
    def test_explain_agrees_with_select(self):
        """The report must never disagree with the behaviour it describes."""
        profile = classify_request("hello")
        opts = [CHEAP_WEAK, STRONG, MID]
        cfg = {"enabled": True}
        decision = select_model(opts, profile, settings=parse_settings(cfg))
        explained = explain_routing(cfg, "hello", options=opts)
        assert explained["selected"] == decision.model

    def test_disabled_is_reported(self):
        assert explain_routing({}, "hi", options=[CHEAP_WEAK])["enabled"] is False

    def test_exclusion_reason_given(self):
        r = explain_routing({"enabled": True}, "describe this",
                            options=[TEXT_ONLY], has_images=True)
        row = r["candidates"][0]
        assert row["eligible"] is False
        assert row["excluded_reason"]

    def test_unpriced_model_reported_as_median(self):
        r = explain_routing({"enabled": True}, "hello",
                            options=[ModelOption(name="free-ish"), CHEAP_WEAK])
        row = next(c for c in r["candidates"] if c["model"] == "free-ish")
        assert row["priced"] is False and row["charged_as"] == "median_unknown"

    def test_unknown_capabilities_listed(self):
        r = explain_routing({"enabled": True}, "hello", options=[ModelOption(name="sparse")])
        assert "tools" in r["candidates"][0]["unknown_capabilities"]

    def test_reports_settings_used(self):
        r = explain_routing({"enabled": True, "strategy": "quality"}, "hi", options=[MID])
        assert r["settings"]["strategy"] == "quality"


class TestSkipReasons:
    """``skip_reason`` is the machine-readable contract; ``reason`` is prose and may be reworded."""

    def test_switch_has_no_skip_reason(self):
        from agent.model_routing import SKIP_ALREADY_CURRENT
        decision = select_model([CHEAP_WEAK, STRONG], classify_request("```refactor```"),
                                current="cheap", min_improvement=0.1)
        assert decision.model == "strong" and decision.skip_reason != SKIP_ALREADY_CURRENT
        assert decision.skip_reason == ""

    def test_no_candidates(self):
        from agent.model_routing import SKIP_NO_CANDIDATES
        assert select_model([], classify_request("hi")).skip_reason == SKIP_NO_CANDIDATES

    def test_no_eligible_model(self):
        from agent.model_routing import SKIP_NO_ELIGIBLE
        profile = classify_request("see image", has_images=True)
        assert select_model([TEXT_ONLY], profile).skip_reason == SKIP_NO_ELIGIBLE

    def test_already_current(self):
        from agent.model_routing import SKIP_ALREADY_CURRENT
        decision = select_model([CHEAP_WEAK, STRONG], classify_request("```refactor```"),
                                current="strong", min_improvement=0.1)
        assert decision.model == "strong"
        assert decision.skip_reason == SKIP_ALREADY_CURRENT

    def test_within_stability_threshold(self):
        from agent.model_routing import SKIP_WITHIN_THRESHOLD
        decision = select_model([CHEAP_WEAK, STRONG], classify_request("hello"),
                                current="strong", min_improvement=0.9)
        assert decision.skip_reason == SKIP_WITHIN_THRESHOLD

    def test_explain_surfaces_skip_reason(self):
        r = explain_routing({"enabled": True}, "hi", options=[], current="m")
        assert r["enabled"] is True and r["skip_reason"] in ("no_candidates", "")

    def test_skip_reasons_are_distinct(self):
        from agent.model_routing import (
            SKIP_ALREADY_CURRENT, SKIP_DISABLED, SKIP_ERROR, SKIP_NO_CANDIDATES,
            SKIP_NO_ELIGIBLE, SKIP_WITHIN_THRESHOLD,
        )
        reasons = [SKIP_DISABLED, SKIP_NO_CANDIDATES, SKIP_NO_ELIGIBLE, SKIP_ALREADY_CURRENT,
                   SKIP_WITHIN_THRESHOLD, SKIP_ERROR]
        assert len(set(reasons)) == len(reasons)

    def test_disabled_reports_disabled_not_a_switch(self):
        from agent.model_routing import SKIP_DISABLED
        r = explain_routing({}, "hi", options=[ModelOption(name="a"), ModelOption(name="b")])
        assert r["enabled"] is False and r["skip_reason"] == SKIP_DISABLED


class TestMoAGuard:
    """Both implementations skip routing while the Mixture-of-Agents facade is active."""

    ENABLED = {"smart_model_routing": {"enabled": True, "models": [
        {"name": "cheap", "cost_in": 0.1, "tier_rank": 0}]}}

    def test_provider_moa_skips_routing(self, monkeypatch):
        calls = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda *a, **k: calls.append(a))
        agent = _Agent(self.ENABLED, model="default", provider="moa")
        assert maybe_apply_model_routing(agent, "hello") is None
        assert calls == []

    def test_requested_provider_moa_is_enough(self, monkeypatch):
        """``/model`` pins requested_provider before provider; either is a skip signal."""
        calls = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda *a, **k: calls.append(a))
        agent = _Agent(self.ENABLED, model="default", provider="p")
        agent.requested_provider = "MoA"
        assert maybe_apply_model_routing(agent, "hello") is None
        assert calls == []

    def test_routing_still_runs_for_a_normal_provider(self, monkeypatch):
        calls = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda *a, **k: calls.append(a))
        agent = _Agent(self.ENABLED, model="strong", provider="p")
        maybe_apply_model_routing(agent, "hello")
        assert calls, "routing must not be disabled outside MoA"


class TestConfigCatalogMerge:
    """``models[]`` is canonical; per-model fields override catalog metadata, absent ones inherit."""

    @staticmethod
    def _info():
        from types import SimpleNamespace
        return SimpleNamespace(
            context_window=200_000, max_output=8_000, cost_input=3.0, cost_output=15.0,
            input_modalities=("text", "image"), output_modalities=("text",),
            tool_call=True, structured_output=True, reasoning=False, status="active")

    def test_config_wins_and_omitted_fields_come_from_the_catalog(self, monkeypatch):
        import agent.models_dev as models_dev
        monkeypatch.setattr(models_dev, "get_model_info", lambda *a, **k: self._info())
        option = options_from_config(
            {"models": [{"name": "m", "provider": "p", "cost_in": 0.5}]})[0]
        assert option.cost_in == 0.5              # stated in config, so config wins
        assert option.cost_out == 15.0            # inherited from the catalog
        assert option.context_window == 200_000
        assert option.supports_vision is True
        assert option.tier_rank == 4              # catalog: tool calling + 200k context

    def test_explicit_false_in_config_beats_the_catalog(self, monkeypatch):
        import agent.models_dev as models_dev
        monkeypatch.setattr(models_dev, "get_model_info", lambda *a, **k: self._info())
        option = options_from_config(
            {"models": [{"name": "m", "provider": "p", "supports_tools": False}]})[0]
        assert option.supports_tools is False

    def test_silent_catalog_leaves_capabilities_unknown(self, monkeypatch):
        import agent.models_dev as models_dev
        monkeypatch.setattr(models_dev, "get_model_info", lambda *a, **k: None)
        option = options_from_config({"models": [{"name": "m", "provider": "p"}]})[0]
        assert option.supports_tools is None and option.supports_vision is None
        assert option.cost_in == 0.0 and option.tier_rank == 3

    def test_no_provider_means_no_catalog_lookup(self, monkeypatch):
        import agent.models_dev as models_dev
        called = []
        monkeypatch.setattr(models_dev, "get_model_info", lambda *a, **k: called.append(a))
        option = options_from_config({"models": [{"name": "ok"}]})[0]
        assert called == []
        assert option.cost_in == 0.0 and option.supports_tools is None


class TestValidatorFindings:
    """Regressions for four defects found by independent validation."""

    def test_preferred_models_only_without_a_list_does_not_route_the_whole_catalog(self, monkeypatch):
        """The setting must not silently do the opposite of its name."""
        calls = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda *a, **k: calls.append(a))
        agent = _Agent({"smart_model_routing": {
            "enabled": True, "preferred_models_only": True, "models": []}},
            model="m", provider="p")
        assert maybe_apply_model_routing(agent, "do something") is None
        assert calls == []

    def test_preferred_models_only_still_routes_within_an_explicit_list(self, monkeypatch):
        calls = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda *a, **k: calls.append(a))
        agent = _Agent(model="expensive", provider="p",
                       config={"smart_model_routing": {
                           "enabled": True, "preferred_models_only": True,
                           "models": [
                               {"name": "expensive", "provider": "p", "cost_in": 5.0,
                                "cost_out": 15.0, "tier_rank": 3},
                               {"name": "cheap", "provider": "p", "cost_in": 0.1,
                                "cost_out": 0.2, "tier_rank": 3}]}})
        decision = maybe_apply_model_routing(agent, "hello")
        assert decision is not None and decision.model == "cheap"

    def test_preferred_models_only_defaults_to_false(self):
        """C6 of docs/routing-contract.md: the restriction is an explicit opt-in."""
        assert parse_settings({}).preferred_models_only is False
        assert parse_settings({"preferred_models_only": True}).preferred_models_only is True

    def test_absent_preferred_models_only_falls_back_to_the_catalog(self, monkeypatch):
        """Default false means no models list still routes over the cached catalog."""
        calls = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda *a, **k: calls.append(a))
        monkeypatch.setattr(model_routing, "options_from_catalog", lambda *a, **k: [
            ModelOption(name="expensive", provider="p", cost_in=5.0, cost_out=15.0, tier_rank=3),
            ModelOption(name="cheap", provider="p", cost_in=0.1, cost_out=0.2, tier_rank=3),
        ])
        agent = _Agent({"smart_model_routing": {"enabled": True, "models": []}},
                       model="expensive", provider="p")
        decision = maybe_apply_model_routing(agent, "hello")
        assert decision is not None and decision.model == "cheap"

    def test_explain_scores_do_not_collide_on_duplicate_names(self):
        a = ModelOption(name="shared", provider="alpha", cost_in=0.1, cost_out=0.1, tier_rank=3)
        b = ModelOption(name="shared", provider="beta", cost_in=9.0, cost_out=9.0, tier_rank=3)
        r = explain_routing({"enabled": True}, "hello", options=[a, b])
        scores = {c["provider"]: c["score"] for c in r["candidates"]}
        assert scores["alpha"] != scores["beta"]

    def test_strict_mode_exclusion_names_audio(self):
        opt = ModelOption(name="no-audio-meta", tier_rank=3, supports_tools=True)
        profile = classify_request("x", has_audio=True)
        assert is_eligible(opt, profile, allow_unknown=False) is False
        assert "audio" in _exclusion_reason(opt, profile, parse_settings(
            {"allow_unknown_metadata": False}))

    def test_option_building_errors_are_swallowed(self, monkeypatch):
        """The docstring promises any failure is swallowed; options building is inside it now."""
        agent = _Agent({"smart_model_routing": {"enabled": True}}, model="m", provider="p")
        monkeypatch.setattr(model_routing, "options_from_config",
                            lambda cfg: (_ for _ in ()).throw(RuntimeError("corrupt cache")))
        assert maybe_apply_model_routing(agent, "hello") is None
