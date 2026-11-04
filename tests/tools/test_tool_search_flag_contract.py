"""Flag contract for the tool_search reliability-ranking switch.

Canonical key: ``reliability_rerank``. ``reliability`` is honoured as a deprecated alias.
The contract these tests pin down:

  (a) canonical present                => canonical decides
  (b) canonical + legacy present       => canonical wins, warning naming both
  (c) only legacy present              => honoured, deprecation warning points at canonical
  (d) neither present                  => False
  (e) unknown key under tools.tool_search => one warning naming it, saying it is NOT in effect
  (f) flag off                         => strict no-op, no reliability store read

The failure mode this guards is silence: a config typo or a pre-rename key that is quietly
dropped looks identical to "off" and costs a debugging session.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_LOG = "tools.tool_search"


def _flag(raw):
    from tools.tool_search import _reliability_flag

    return _reliability_flag(raw)


def _warnings(caplog):
    """Only this module's WARNING+ records — unrelated libraries also log."""
    return [r.getMessage() for r in caplog.records
            if r.name == _LOG and r.levelno >= logging.WARNING]


@pytest.fixture
def caplog_flag(caplog):
    caplog.set_level(logging.WARNING, logger=_LOG)
    return caplog


# ---------------------------------------------------------------------------
# Key names
# ---------------------------------------------------------------------------


def test_flag_names_are_canonical_and_legacy():
    from tools.tool_search import RELIABILITY_FLAG, RELIABILITY_FLAG_LEGACY

    assert RELIABILITY_FLAG == "reliability_rerank"
    assert RELIABILITY_FLAG_LEGACY == "reliability"


# ---------------------------------------------------------------------------
# (d) absent => False
# ---------------------------------------------------------------------------


def test_absent_flag_is_false(caplog_flag):
    assert _flag({}) is False
    assert _flag({"enabled": "on"}) is False
    assert _warnings(caplog_flag) == []


# ---------------------------------------------------------------------------
# (a) canonical present => canonical decides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, "true", "TRUE", "on", "1", "yes", " on "])
def test_canonical_true_forms_enable(value, caplog_flag):
    from tools.tool_search import ToolSearchConfig

    assert _flag({"reliability_rerank": value}) is True
    assert ToolSearchConfig.from_raw({"reliability_rerank": value}).reliability is True
    assert _warnings(caplog_flag) == []


@pytest.mark.parametrize("value", [False, "false", "off", "0", "no", "", None])
def test_canonical_false_forms_disable(value, caplog_flag):
    from tools.tool_search import ToolSearchConfig

    assert _flag({"reliability_rerank": value}) is False
    assert ToolSearchConfig.from_raw({"reliability_rerank": value}).reliability is False
    assert _warnings(caplog_flag) == []


# ---------------------------------------------------------------------------
# (c) legacy alone => honoured + deprecation warning
# ---------------------------------------------------------------------------


def test_legacy_alias_true_is_honoured_with_deprecation_warning(caplog_flag):
    from tools.tool_search import ToolSearchConfig

    cfg = ToolSearchConfig.from_raw({"reliability": True})
    assert cfg.reliability is True
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1
    assert "deprecated" in warnings[0].lower()
    assert "reliability_rerank" in warnings[0]
    assert "reliability" in warnings[0]


def test_legacy_alias_false_is_honoured_and_still_warns(caplog_flag):
    assert _flag({"reliability": "off"}) is False
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1
    assert "deprecated" in warnings[0].lower()
    assert "reliability_rerank" in warnings[0]


def test_legacy_alias_does_not_count_as_unknown_key(caplog_flag):
    _flag({"reliability": True})
    assert all("unknown" not in w.lower() for w in _warnings(caplog_flag))


# ---------------------------------------------------------------------------
# (b) both present => canonical wins, warning names both
# ---------------------------------------------------------------------------


def test_canonical_wins_conflict(caplog_flag):
    assert _flag({"reliability_rerank": True, "reliability": False}) is True
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1
    assert "reliability_rerank" in warnings[0] and "reliability" in warnings[0]


def test_canonical_off_beats_alias_on(caplog_flag):
    from tools.tool_search import ToolSearchConfig

    cfg = ToolSearchConfig.from_raw({"reliability_rerank": "false", "reliability": "true"})
    assert cfg.reliability is False
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1
    assert "reliability_rerank" in warnings[0] and "reliability" in warnings[0]


def test_canonical_garbage_does_not_fall_back_to_alias(caplog_flag):
    """Presence decides, not validity — a broken canonical value must not resurrect the alias."""
    assert _flag({"reliability_rerank": "maybe", "reliability": True}) is False
    assert len(_warnings(caplog_flag)) == 1


# ---------------------------------------------------------------------------
# (e) unknown keys are reported, never silently ignored
# ---------------------------------------------------------------------------


def test_unknown_key_is_reported(caplog_flag):
    from tools.tool_search import ToolSearchConfig

    cfg = ToolSearchConfig.from_raw({"enabled": "on", "reliabilty_rerank": True})
    assert cfg.reliability is False  # typo is NOT in effect
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1
    assert "reliabilty_rerank" in warnings[0]
    assert "not in effect" in warnings[0].lower()


def test_unknown_keys_are_listed_in_one_warning(caplog_flag):
    from tools.tool_search import ToolSearchConfig

    ToolSearchConfig.from_raw({"bogus_a": 1, "bogus_b": 2})
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1
    assert "bogus_a" in warnings[0] and "bogus_b" in warnings[0]


def test_unknown_key_does_not_disturb_known_keys(caplog_flag):
    from tools.tool_search import ToolSearchConfig

    cfg = ToolSearchConfig.from_raw({
        "reliability_rerank": True, "threshold_pct": 12, "search_default_limit": 7,
        "legacy_noise": "x",
    })
    assert cfg.reliability is True
    assert cfg.threshold_pct == 12.0
    assert cfg.search_default_limit == 7
    warnings = _warnings(caplog_flag)
    assert len(warnings) == 1 and "legacy_noise" in warnings[0]


# ---------------------------------------------------------------------------
# clean config => no warnings at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [
    None, False, True,
    {},
    {"enabled": "auto"},
    {"reliability_rerank": True},
    {"enabled": "on", "listing": "auto", "defer": ["todo_list"], "zero_match_llm": False},
])
def test_clean_config_logs_nothing(raw, caplog_flag):
    from tools.tool_search import ToolSearchConfig

    ToolSearchConfig.from_raw(raw)
    assert _warnings(caplog_flag) == []


def test_shipped_defaults_use_canonical_key_and_are_silent(caplog_flag):
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.tool_search import ToolSearchConfig

    section = DEFAULT_CONFIG["tools"]["tool_search"]
    assert "reliability_rerank" in section
    assert "reliability" not in section  # a shipped alias would warn on every load
    assert ToolSearchConfig.from_raw(section).reliability is False
    assert _warnings(caplog_flag) == []


# ---------------------------------------------------------------------------
# alias-on is behaviourally identical to canonical-on
# ---------------------------------------------------------------------------


def test_alias_on_matches_canonical_on(caplog_flag):
    from tools.tool_search import ToolSearchConfig

    canonical = ToolSearchConfig.from_raw({"reliability_rerank": True, "threshold_pct": 8})
    alias = ToolSearchConfig.from_raw({"reliability": True, "threshold_pct": 8})
    assert canonical == alias  # frozen dataclass: field-for-field equality
    assert alias.reliability is canonical.reliability is True


# ---------------------------------------------------------------------------
# (f) flag off stays a strict no-op: no reliability store read
# ---------------------------------------------------------------------------


def _counting_store(monkeypatch):
    import tools.tool_search as tool_search

    calls = []
    monkeypatch.setattr(tool_search, "load_tool_reliability",
                        lambda *a, **kw: calls.append((a, kw)) or {})
    return calls


def test_config_parse_never_reads_the_store(monkeypatch):
    from tools.tool_search import ToolSearchConfig

    calls = _counting_store(monkeypatch)
    ToolSearchConfig.from_raw({"reliability_rerank": True})
    ToolSearchConfig.from_raw({"reliability": True})
    assert calls == []


def test_disabled_flag_reads_no_store(monkeypatch):
    import tools.tool_search as tool_search

    calls = _counting_store(monkeypatch)
    cfg = tool_search.ToolSearchConfig.from_raw({"reliability_rerank": False})
    payload = json.loads(tool_search.dispatch_tool_search(
        {"queries": ["anything"]}, current_tool_defs=[], config=cfg))
    assert payload["results"][0]["matches"] == []
    assert calls == []


def test_enabled_flag_reads_the_store_once_per_dispatch(monkeypatch):
    import tools.tool_search as tool_search

    calls = _counting_store(monkeypatch)
    cfg = tool_search.ToolSearchConfig.from_raw({"reliability_rerank": True})
    tool_search.dispatch_tool_search(
        {"queries": ["one", "two", "three"]}, current_tool_defs=[], config=cfg)
    assert len(calls) == 1  # once per call, never per query
