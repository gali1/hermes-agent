"""Configuration contract: one canonical reliability flag across both deployments.

``reliability_rerank`` is canonical; ``reliability`` is the pre-unification name and keeps
working through an explicit alias. The rules under test:

* the canonical key decides, alone;
* the legacy alias is honoured AND announced as deprecated — never silently dropped;
* when both are present the canonical key wins and the conflict is reported;
* an unknown key is reported as "not in effect", not swallowed;
* alias-on and canonical-on produce the same behaviour, so a config moved between the
  two hosts cannot change what the agent does.
"""

import logging

import pytest


def _cfg(raw):
    from tools.tool_search import ToolSearchConfig

    return ToolSearchConfig.from_raw(raw)


class TestCanonicalKey:
    def test_absent_is_off(self):
        assert _cfg({}).reliability_rerank is False
        assert _cfg(None).reliability_rerank is False

    @pytest.mark.parametrize("value", [True, "true", "on", "1", "yes", 1])
    def test_true_forms_enable(self, value):
        assert _cfg({"reliability_rerank": value}).reliability_rerank is True

    @pytest.mark.parametrize("value", [False, "false", "off", "0", "no", 0, "", None, "typo"])
    def test_false_forms_stay_off(self, value):
        assert _cfg({"reliability_rerank": value}).reliability_rerank is False


class TestLegacyAlias:
    def test_alias_is_honoured(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            cfg = _cfg({"reliability": "true"})
        assert cfg.reliability_rerank is True
        assert any("deprecated" in r.message and "reliability_rerank" in r.message
                   for r in caplog.records), [r.message for r in caplog.records]

    def test_alias_false_does_not_enable(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            cfg = _cfg({"reliability": "false"})
        assert cfg.reliability_rerank is False

    def test_canonical_wins_and_conflict_is_reported(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            cfg = _cfg({"reliability_rerank": True, "reliability": "false"})
        assert cfg.reliability_rerank is True          # canonical decides
        assert any("both" in r.message.lower() for r in caplog.records), \
            [r.message for r in caplog.records]

    def test_canonical_off_beats_alias_on(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            cfg = _cfg({"reliability_rerank": False, "reliability": "true"})
        assert cfg.reliability_rerank is False


class TestUnknownKeys:
    def test_unknown_key_is_reported_not_swallowed(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            cfg = _cfg({"reliability_re_rank": True})   # plausible typo
        assert cfg.reliability_rerank is False          # not in effect
        messages = [r.message for r in caplog.records]
        assert any("unknown key" in m and "reliability_re_rank" in m for m in messages), messages
        assert any("NOT in effect" in m for m in messages), messages

    def test_unknown_key_does_not_disturb_known_ones(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            cfg = _cfg({"zero_match_llm": "on", "nonsense": 1})
        assert cfg.zero_match_llm is True
        assert cfg.zero_match_llm_timeout == 4.0

    def test_no_warning_for_a_clean_config(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            _cfg({"enabled": "auto", "reliability_rerank": True, "zero_match_llm": True})
        assert not [r for r in caplog.records if "unknown key" in r.message]


class TestBehaviourEquivalence:
    """The alias must be a rename, not a second code path."""

    def _tie_catalog(self):
        from tests.tools.test_tool_search_anytool import _entry

        return [_entry("anytool_alpha_widget_lookup", "Look up a widget by id"),
                _entry("anytool_beta_widget_lookup", "Look up a widget by id")]

    def _order(self, multipliers):
        from tools.tool_search_catalog import search_catalog

        return [e.name for e in search_catalog(
            self._tie_catalog(), "widget lookup", limit=5, multipliers=multipliers)]

    def test_alias_on_equals_canonical_on(self, caplog):
        from tools.tool_search import ToolSearchConfig

        with caplog.at_level(logging.WARNING, logger="tools.tool_search"):
            legacy = ToolSearchConfig.from_raw({"reliability": "on"})
        canonical = ToolSearchConfig.from_raw({"reliability_rerank": "on"})
        assert legacy.reliability_rerank == canonical.reliability_rerank is True

        multipliers = {"anytool_alpha_widget_lookup": 0.2}
        assert self._order(multipliers) == ["anytool_beta_widget_lookup",
                                            "anytool_alpha_widget_lookup"]

    def test_disabled_still_reads_no_store(self, monkeypatch):
        """Both spellings off => the loader is never called (strict no-op preserved)."""
        import tools.tool_search as ts

        calls = []

        def _boom(*args, **kwargs):
            calls.append(1)
            raise AssertionError("outcome store must not be read when the flag is off")

        monkeypatch.setattr(ts, "load_reliability_multipliers", _boom, raising=False)
        for raw in ({}, {"reliability_rerank": False}, {"reliability": "false"}):
            assert ts._reliability_multipliers(ts.ToolSearchConfig.from_raw(raw)) is None
        assert calls == []
