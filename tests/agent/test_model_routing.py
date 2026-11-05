"""Tests for the deterministic model-routing engine.

The engine is arithmetic, so these tests are exact: no fakes, no clock, no network. The two
properties that matter most are that an ineligible model can never win on price, and that a
disabled config is a strict no-op.
"""

from __future__ import annotations

import statistics

import pytest

import agent.model_routing as model_routing
from agent.model_routing import (
    TIERS, ModelOption, RequestProfile, classify_request, estimate_cost, estimate_tokens,
    _exclusion_reason, explain_routing, is_eligible, maybe_apply_model_routing,
    options_from_config, parse_settings, rank_candidates, score_model, select_model,
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


class TestMultimodalPayload:
    """A content-parts list must not raise, and must not be read as prompt text.

    It used to raise ``AttributeError: 'list' object has no attribute 'strip'`` inside a
    swallowed ``except``, so every multimodal turn silently kept the current model.
    """

    def test_content_parts_list_does_not_raise(self):
        parts = [{"type": "text", "text": "hello"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        profile = classify_request(parts)
        assert profile.tier == "specialized"
        assert profile.needs_vision is True
        # The image payload is a modality, not prose: it must not inflate the token estimate.
        assert profile.est_input_tokens < 20

    def test_audio_part_sets_audio_requirement(self):
        assert classify_request([{"type": "input_audio", "input_audio": {"data": "AA"}}]).needs_audio

    def test_file_part_sets_file_requirement(self):
        profile = classify_request([{"type": "file", "file": {"file_id": "f1"}}])
        assert profile.tier == "specialized" and profile.reasons == ("modality required",)

    def test_multimodal_envelope_is_unwrapped(self):
        envelope = {"_multimodal": True, "content": [
            {"type": "text", "text": "describe"},
            {"type": "input_image", "image_url": "x"}]}
        profile = classify_request(envelope)
        assert profile.needs_vision is True and profile.tier == "specialized"

    def test_explicit_flag_survives_parts(self):
        assert classify_request([{"type": "text", "text": "hi"}], has_images=True).needs_vision

    def test_string_prompt_with_image_path_requires_vision(self, tmp_path):
        """C9a for the string path: an attached image is a hard requirement, not prose.

        ``image_routing.extract_image_refs`` is what later turns the path into a content part,
        so routing must ask the same question here or a text-only model wins the turn and the
        attachment is dropped/failed after selection.
        """
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        profile = classify_request(f"what is wrong in {img}?")
        assert profile.needs_vision is True and profile.tier == "specialized"
        assert not is_eligible(TEXT_ONLY, profile)
        assert is_eligible(VISION, profile)

    def test_string_prompt_with_missing_path_does_not_require_vision(self):
        """A path that is not a file is prose, not an attachment."""
        profile = classify_request("look at /nope/definitely-missing.png and tell me")
        assert profile.needs_vision is False

    def test_explain_agrees_with_selection_on_string_image_path(self, tmp_path):
        """The explain surface must not re-derive modality differently from selection."""
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        report = explain_routing({"enabled": True}, f"describe {img}", options=[TEXT_ONLY, VISION])
        assert report["request"]["needs_vision"] is True
        rows = {row["model"]: row for row in report["candidates"]}
        assert rows["text-only"]["eligible"] is False
        assert "vision" in rows["text-only"]["excluded_reason"]
        assert rows["vision"]["eligible"] is True


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


class TestIncumbentIdentity:
    """C5: the incumbent is ``(provider, model)``; name-only is only for callers without one."""

    CHEAP_DUP = ModelOption(name="dup", provider="a", cost_in=1.0, cost_out=1.0, tier_rank=3)
    PRICEY_DUP = ModelOption(name="dup", provider="b", cost_in=100.0, cost_out=100.0, tier_rank=3)

    @staticmethod
    def _settings():
        return parse_settings({"enabled": True, "strategy": "cost"})

    def test_same_name_other_provider_is_not_the_incumbent(self):
        """A move to an identically named model on another provider must stay reachable."""
        decision = select_model([self.CHEAP_DUP, self.PRICEY_DUP], classify_request("hello"),
                                current="dup", current_provider="b", settings=self._settings())
        assert (decision.model, decision.provider) == ("dup", "a")
        assert decision.skip_reason == ""

    def test_incumbent_on_its_own_provider_is_reported_with_that_provider(self):
        decision = select_model([self.CHEAP_DUP, self.PRICEY_DUP], classify_request("hello"),
                                current="dup", current_provider="a", settings=self._settings())
        assert decision.provider == "a"
        assert decision.skip_reason == "already_current"

    def test_no_provider_falls_back_to_name_only(self):
        decision = select_model([self.CHEAP_DUP, self.PRICEY_DUP], classify_request("hello"),
                                current="dup", settings=self._settings())
        assert decision.skip_reason == "already_current" and decision.provider == "a"

    def test_wiring_switches_provider_for_identically_named_model(self, monkeypatch):
        """The early \"unchanged\" return must compare (provider, model), not the name."""
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p: switched.append((m, p)))
        agent = _Agent({"smart_model_routing": {
            "enabled": True, "strategy": "cost",
            "models": [
                {"name": "dup", "provider": "a", "cost_in": 1.0, "cost_out": 1.0, "tier_rank": 3},
                {"name": "dup", "provider": "b", "cost_in": 100.0, "cost_out": 100.0,
                 "tier_rank": 3},
            ]}}, model="dup", provider="b")
        decision = maybe_apply_model_routing(agent, "hello")
        assert decision is not None and decision.provider == "a"
        assert switched == [("dup", "a")]


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
        agent = _Agent({"smart_model_routing": {"enabled": True, "preferred_models_only": False,
                                                "models": [
            {"name": "cheap", "cost_in": 0.1, "tier_rank": 0}]}}, model="strong")

        def _boom(*a, **k):
            raise RuntimeError("switch exploded")

        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", _boom)
        assert maybe_apply_model_routing(agent, "hello") is None

    def test_multimodal_list_routes_instead_of_going_inert(self, monkeypatch):
        """An image part must not be swallowed by the exception handler (C9a, bro-2 side)."""
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p: switched.append((m, p)))
        agent = _Agent({"smart_model_routing": {"enabled": True, "models": [
            {"name": "text-only", "provider": "p", "cost_in": 0.1, "cost_out": 0.1,
             "tier_rank": 5, "supports_vision": False},
            {"name": "vision", "provider": "p", "cost_in": 1.0, "cost_out": 1.0,
             "tier_rank": 5, "supports_vision": True},
        ]}}, model="text-only", provider="p")
        parts = [{"type": "text", "text": "what is this"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        decision = maybe_apply_model_routing(agent, parts)
        assert decision is not None and decision.model == "vision"
        assert switched == [("vision", "p")]

    def test_single_model_pool_is_noop_without_explicit_key(self, monkeypatch):
        """One preferred model is not a choice: no catalog scan, no routing decision."""
        catalog_calls = []

        def _catalog(provider=""):
            catalog_calls.append(provider)
            return []

        def _no_switch(*a, **k):
            raise AssertionError("a single-model pool must not route")

        monkeypatch.setattr(model_routing, "options_from_catalog", _catalog)
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", _no_switch)
        agent = _Agent({"smart_model_routing": {"enabled": True, "models": [
            {"name": "cheap", "cost_in": 0.1, "cost_out": 0.2, "tier_rank": 0}]}},
            model="strong")
        assert maybe_apply_model_routing(agent, "hello") is None
        assert catalog_calls == []

    def test_explicit_false_lets_routing_run(self, monkeypatch):
        """Writing the key turns the single-model no-op back into a real decision."""
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p: switched.append((m, p)))
        agent = _Agent({"smart_model_routing": {
            "enabled": True, "preferred_models_only": False,
            "models": [{"name": "cheap", "cost_in": 0.1, "cost_out": 0.2, "tier_rank": 0}]}},
            model="strong")
        decision = maybe_apply_model_routing(agent, "hello")
        assert decision is not None and decision.model == "cheap"
        assert switched == [("cheap", "p")]


class TestSettingsContract:
    """One authoritative smart_model_routing contract, both historical spellings accepted."""

    def test_defaults(self):
        st = parse_settings({})
        assert st.enabled is False and st.strategy == "automatic"
        assert st.stability_threshold == 0.15 and st.allow_unknown_metadata is True

    def test_preferred_models_only_default_is_false_and_key_presence_tracked(self):
        st = parse_settings({})
        assert st.preferred_models_only is False
        assert st.preferred_models_only_explicit is False
        assert parse_settings({"preferred_models_only": False}).preferred_models_only_explicit is True
        written = parse_settings({"preferred_models_only": True})
        assert written.preferred_models_only is True and written.preferred_models_only_explicit

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

    def test_excluded_reason_is_empty_string_when_eligible(self):
        """One row per candidate, and the field is always a string (C10)."""
        r = explain_routing({"enabled": True}, "hello", options=[CHEAP_WEAK])
        assert r["candidates"][0]["excluded_reason"] == ""

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

    ENABLED = {"smart_model_routing": {"enabled": True, "preferred_models_only": False,
                                      "models": [
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


class TestExpectedTotalCost:
    """§7.4: rank on the expected TOTAL cost, not the price of one happy-path attempt.

    Three terms were missing: cached-input pricing (the catalog publishes ``cost.cache_read``
    and ``cost.cache_write``), reasoning tokens (billed at the output rate — models.dev has no
    reasoning price field, so none was invented) and expected retries.
    """

    PROFILE = RequestProfile("routine", est_input_tokens=100_000, est_output_tokens=1_000)

    def test_defaults_keep_the_old_one_attempt_formula(self):
        """Cache prices on the model must not change the number unless a caller asks."""
        option = ModelOption(name="m", cost_in=1.0, cost_out=2.0,
                             cost_cache_read=0.01, cost_cache_write=0.02)
        assert estimate_cost(option, RequestProfile("routine", 1_000, 1_000)) == pytest.approx(0.003)

    def test_cached_input_price_reorders_two_models(self):
        cached = ModelOption(name="cached", cost_in=5.0, cost_out=1.0, cost_cache_read=0.05)
        plain = ModelOption(name="plain", cost_in=1.0, cost_out=1.0)
        # One uncached attempt: the plain model is much cheaper.
        assert estimate_cost(plain, self.PROFILE) < estimate_cost(cached, self.PROFILE)
        # With 90% of the prompt served from cache, the cache-friendly model wins.
        assert estimate_cost(cached, self.PROFILE, cached_input_ratio=0.9) < estimate_cost(
            plain, self.PROFILE, cached_input_ratio=0.9)

    def test_cached_tokens_without_a_published_cache_price_are_not_free(self):
        """A silent cache price falls back to the input rate; it must never read as zero."""
        option = ModelOption(name="m", cost_in=1.0, cost_out=2.0)
        assert estimate_cost(option, self.PROFILE, cached_input_ratio=1.0) == pytest.approx(
            estimate_cost(option, self.PROFILE))

    def test_cache_write_is_priced_at_its_own_rate(self):
        option = ModelOption(name="m", cost_in=1.0, cost_out=0.0, cost_cache_write=2.0)
        assert estimate_cost(option, self.PROFILE, cache_write_ratio=0.5) == pytest.approx(
            (50_000 * 1.0 + 50_000 * 2.0) / 1_000_000.0)

    def test_overlapping_cache_shares_cannot_bill_more_than_the_prompt(self):
        option = ModelOption(name="m", cost_in=10.0, cost_out=0.0, cost_cache_read=0.0,
                             cost_cache_write=0.0)
        assert estimate_cost(option, self.PROFILE, cached_input_ratio=1.0,
                             cache_write_ratio=1.0) == pytest.approx(0.0)

    def test_reasoning_tokens_are_charged_at_the_output_rate(self):
        """No reasoning price exists in the catalog, so thinking is billed as output."""
        option = ModelOption(name="m", cost_in=1.0, cost_out=10.0)
        base = estimate_cost(option, self.PROFILE)
        with_reasoning = estimate_cost(option, self.PROFILE, reasoning_tokens=2_000)
        assert with_reasoning - base == pytest.approx(2_000 * 10.0 / 1_000_000.0)

    def test_reasoning_tokens_flip_the_ranking_toward_a_lower_output_rate(self):
        flat = ModelOption(name="flat", cost_in=0.01, cost_out=2.0, tier_rank=3)
        premium = ModelOption(name="premium", cost_in=10.0, cost_out=0.5, tier_rank=3)
        base = RequestProfile("routine", est_input_tokens=1_000, est_output_tokens=100)
        thinking = RequestProfile("routine", est_input_tokens=1_000, est_output_tokens=100,
                                  reasoning_tokens=50_000)
        assert estimate_cost(flat, base) < estimate_cost(premium, base)
        assert estimate_cost(premium, thinking) < estimate_cost(flat, thinking)

    def test_a_retry_reuses_the_prompt_cache(self):
        option = ModelOption(name="m", cost_in=10.0, cost_out=1.0, cost_cache_read=0.1)
        once = estimate_cost(option, self.PROFILE)
        twice = estimate_cost(option, self.PROFILE, expected_retries=1)
        retry = (100_000 * 0.1 + 1_000 * 1.0) / 1_000_000.0
        assert twice - once == pytest.approx(retry)
        # A retry is not a second full attempt: the prompt is a cache hit.
        assert 0 < twice - once < once

    def test_profile_assumptions_apply_when_arguments_are_omitted(self):
        option = ModelOption(name="m", cost_in=5.0, cost_out=1.0, cost_cache_read=0.05)
        profile = RequestProfile("routine", est_input_tokens=100_000, est_output_tokens=1_000,
                                 cached_input_ratio=0.9, expected_retries=1)
        assert estimate_cost(option, profile) == pytest.approx(
            estimate_cost(option, profile, cached_input_ratio=0.9, expected_retries=1))
        assert estimate_cost(option, profile) < estimate_cost(
            option, RequestProfile("routine", 100_000, 1_000))

    def test_unpriced_model_is_still_charged_the_median_of_expected_costs(self):
        """The §7.4 extension must not weaken the "median, never zero" invariant."""
        profile = RequestProfile("routine", est_input_tokens=100_000, est_output_tokens=1_000,
                                 cached_input_ratio=0.9)
        unknown = ModelOption(name="unknown")           # publishes no price at all
        cheap = ModelOption(name="cheap", cost_in=5.0, cost_out=1.0, cost_cache_read=0.05)
        pricey = ModelOption(name="pricey", cost_in=9.0, cost_out=9.0)
        ranked, cheapest, median = rank_candidates([unknown, cheap, pricey], profile,
                                                   parse_settings({}))
        scores = {o.name: s for s, o in ranked}
        expected = [estimate_cost(cheap, profile), estimate_cost(pricey, profile)]
        assert cheapest == pytest.approx(min(expected))
        assert median == pytest.approx(statistics.median(expected))
        assert median > 0
        # Charged at the median, the unpriced model loses to the genuinely cheaper one.
        assert scores["unknown"] < scores["cheap"]

    def test_catalog_cache_prices_reach_the_option(self, monkeypatch):
        from types import SimpleNamespace
        import agent.models_dev as models_dev
        info = SimpleNamespace(cost_input=1.0, cost_output=2.0, cost_cache_read=0.1,
                               cost_cache_write=0.5, context_window=100_000, max_output=8_000,
                               input_modalities=("text",), output_modalities=("text",),
                               tool_call=True, structured_output=False, reasoning=False,
                               status="active")
        monkeypatch.setattr(models_dev, "get_model_info", lambda *a, **k: info)
        option = options_from_config({"models": [{"name": "m", "provider": "p"}]})[0]
        assert option.cost_cache_read == 0.1 and option.cost_cache_write == 0.5


class TestToolFallback:
    """§7.7: a preferred list must not block a tool-capable model when the operator allows it."""

    PREFERRED = [ModelOption(name="chat-only", provider="p", cost_in=0.1, cost_out=0.2,
                             tier_rank=3, supports_tools=False)]
    CATALOG = [
        ModelOption(name="tool-pricey", provider="p", cost_in=5.0, cost_out=15.0,
                    tier_rank=3, supports_tools=True),
        ModelOption(name="tool-cheap", provider="p", cost_in=1.0, cost_out=2.0,
                    tier_rank=3, supports_tools=True),
    ]
    TOOL_PROFILE = classify_request("use the shell to list the files", needs_tools=True)
    NO_TOOL_PROFILE = classify_request("hello")

    def test_without_the_opt_in_the_preferred_list_still_blocks(self):
        """Unchanged behaviour: opt-in is what widens, and the default is off."""
        st = parse_settings({"preferred_models_only": True})
        assert st.allow_tool_fallback is False
        decision = select_model(self.PREFERRED, self.TOOL_PROFILE, settings=st,
                                fallback_options=self.CATALOG)
        assert decision.model is None
        assert decision.skip_reason == "no_eligible_model" and decision.candidates == 0

    def test_opt_in_widens_and_picks_the_cheapest_tool_capable_model(self):
        st = parse_settings({"preferred_models_only": True, "allow_tool_fallback": True})
        decision = select_model(self.PREFERRED, self.TOOL_PROFILE, settings=st,
                                fallback_options=self.CATALOG)
        assert decision.model == "tool-cheap"
        assert decision.skip_reason == ""            # a real switch, not a skip
        assert "tool fallback" in decision.reason

    def test_rank_candidates_widens_the_same_way(self):
        st = parse_settings({"preferred_models_only": True, "allow_tool_fallback": True})
        ranked, _, _ = rank_candidates(self.PREFERRED, self.TOOL_PROFILE, st,
                                       fallback_options=self.CATALOG)
        assert [o.name for _, o in ranked] == ["tool-cheap", "tool-pricey"]

    def test_no_widening_when_the_request_needs_no_tools(self):
        st = parse_settings({"preferred_models_only": True, "allow_tool_fallback": True})
        decision = select_model(self.PREFERRED, self.NO_TOOL_PROFILE, settings=st,
                                fallback_options=self.CATALOG)
        assert decision.model == "chat-only"

    def test_no_widening_when_a_preferred_model_already_calls_tools(self):
        st = parse_settings({"preferred_models_only": True, "allow_tool_fallback": True})
        preferred = [ModelOption(name="pref-tools", provider="p", cost_in=5.0, cost_out=15.0,
                                 tier_rank=3, supports_tools=True)]
        decision = select_model(preferred, self.TOOL_PROFILE, settings=st,
                                fallback_options=self.CATALOG)
        assert decision.model == "pref-tools"
        assert "tool fallback" not in decision.reason

    def test_unknown_tool_support_is_not_capability(self):
        """The check reads the real flag: ``None`` is not evidence a model can call tools."""
        st = parse_settings({"preferred_models_only": True, "allow_tool_fallback": True})
        preferred = [ModelOption(name="unknown-tools", provider="p", cost_in=0.1, cost_out=0.2,
                                 tier_rank=3)]
        decision = select_model(preferred, self.TOOL_PROFILE, settings=st,
                                fallback_options=self.CATALOG)
        assert decision.model == "tool-cheap"

    def test_no_widening_without_a_catalog_pool(self):
        st = parse_settings({"preferred_models_only": True, "allow_tool_fallback": True})
        decision = select_model(self.PREFERRED, self.TOOL_PROFILE, settings=st)
        assert decision.model is None and decision.skip_reason == "no_eligible_model"

    def test_setting_defaults_and_validation(self):
        from agent.model_routing import validate_routing_config
        assert parse_settings({}).allow_tool_fallback is False
        assert parse_settings({"allow_tool_fallback": True}).allow_tool_fallback is True
        assert validate_routing_config({"allow_tool_fallback": "yes"}) == [
            "allow_tool_fallback must be true or false"]
        assert validate_routing_config({"allow_tool_fallback": True}) == []

    def test_wiring_widens_to_the_catalog_when_the_opt_in_is_set(self, monkeypatch):
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p: switched.append((m, p)))
        monkeypatch.setattr(model_routing, "options_from_catalog",
                            lambda *a, **k: list(self.CATALOG))
        agent = _Agent({"smart_model_routing": {
            "enabled": True, "preferred_models_only": True, "allow_tool_fallback": True,
            "models": [{"name": "chat-only", "provider": "p", "cost_in": 0.1, "cost_out": 0.2,
                        "tier_rank": 3, "supports_tools": False}]}},
            model="chat-only", provider="p")
        decision = maybe_apply_model_routing(agent, "use the shell to list the files",
                                             needs_tools=True)
        assert decision is not None and decision.model == "tool-cheap"
        assert switched == [("tool-cheap", "p")]

    def test_wiring_does_not_touch_the_catalog_without_the_opt_in(self, monkeypatch):
        calls = []
        monkeypatch.setattr(model_routing, "options_from_catalog",
                            lambda *a, **k: calls.append(a) or [])
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *a, **k: None)
        agent = _Agent({"smart_model_routing": {
            "enabled": True, "preferred_models_only": True,
            "models": [{"name": "chat-only", "provider": "p", "cost_in": 0.1, "cost_out": 0.2,
                        "tier_rank": 3, "supports_tools": False}]}},
            model="chat-only", provider="p")
        decision = maybe_apply_model_routing(agent, "use the shell", needs_tools=True)
        assert decision is not None and decision.model is None
        assert calls == []

    def test_explain_reports_the_widened_pool(self, monkeypatch):
        """The report must not describe a pool the engine would not have used."""
        monkeypatch.setattr(model_routing, "options_from_catalog",
                            lambda *a, **k: list(self.CATALOG))
        cfg = {"enabled": True, "preferred_models_only": True, "allow_tool_fallback": True,
               "models": [{"name": "chat-only", "provider": "p", "cost_in": 0.1,
                           "cost_out": 0.2, "tier_rank": 3, "supports_tools": False}]}
        report = explain_routing(cfg, "use the shell", provider="p", needs_tools=True)
        assert report["widened_for_tools"] is True
        assert report["selected"] == "tool-cheap"
        assert report["settings"]["allow_tool_fallback"] is True
        assert "tool-cheap" in {c["model"] for c in report["candidates"]}
