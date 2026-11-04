"""Tests for the state-changing routing subcommands and the shared config layer.

The point of these tests is that every interface (CLI, wizard, gateway) writes through one
path: read → merge → validate → preview → confirm → atomic write → read back. They assert on
the file that lands on disk, not on the messages, so a change in wording cannot mask a change
in behaviour.
"""

from __future__ import annotations

import argparse

import pytest

from agent.model_routing import merge_routing_config, validate_routing_config
from hermes_cli.subcommands.model import (
    cmd_model_routing_configure,
    cmd_model_routing_disable,
    cmd_model_routing_enable,
)


def _args(**kw):
    base = {"yes": True}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def cfgfile(tmp_path, monkeypatch):
    """Point the CLI's config path at a scratch file; the real read/write functions run."""
    import hermes_cli.config as config_mod

    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: deepseek-v4.1-flash\n  provider: opencode-zen\nweb:\n  backend: searxng\n")
    monkeypatch.setattr(config_mod, "get_config_path", lambda: path)
    return path


# --- shared layer -----------------------------------------------------------------------

def test_validate_accepts_a_good_block():
    assert validate_routing_config({"enabled": True, "strategy": "cost", "stability_threshold": 0.2,
                                    "models": ["a"]}) == []


@pytest.mark.parametrize("block,needle", [
    ({"strategy": "cheapest"}, "strategy must be one of"),
    ({"stability_threshold": 5}, "between 0 and 1"),
    ({"stability_threshold": "abc"}, "must be a number"),
    ({"models": "gpt-5-nano"}, "must be a list"),
    ({"enabled": "yes"}, "must be true or false"),
    (["not", "a", "mapping"], "must be a mapping"),
])
def test_validate_rejects_bad_values(block, needle):
    problems = validate_routing_config(block)
    assert problems and any(needle in p for p in problems)


def test_merge_preserves_keys_it_does_not_own():
    """A key this engine does not know must survive a write from any interface."""
    merged = merge_routing_config({"enabled": False, "future_knob": "keep-me"}, enabled=True)
    assert merged["enabled"] is True
    assert merged["future_knob"] == "keep-me"


def test_merge_ignores_none_changes():
    merged = merge_routing_config({"enabled": True}, strategy=None)
    assert merged == {"enabled": True}


# --- CLI --------------------------------------------------------------------------------

def test_enable_writes_true_and_leaves_other_sections(cfgfile, capsys):
    assert cmd_model_routing_enable(_args()) == 0
    text = cfgfile.read_text()
    assert "smart_model_routing:" in text and "enabled: true" in text
    assert "backend: searxng" in text          # unrelated key survives
    assert "read back from disk" in capsys.readouterr().out


def test_disable_writes_false(cfgfile):
    assert cmd_model_routing_enable(_args()) == 0
    assert cmd_model_routing_disable(_args()) == 0
    assert "enabled: false" in cfgfile.read_text()


def test_configure_sets_models_and_strategy(cfgfile):
    args = _args(strategy="cost", threshold=0.3, models="gpt-5-nano,claude-haiku-5-5",
                 all_eligible=False, selected_only=True, allow_unknown=None, cost_opt=None)
    assert cmd_model_routing_configure(args) == 0
    text = cfgfile.read_text()
    assert "strategy: cost" in text
    assert "stability_threshold: 0.3" in text
    assert "- gpt-5-nano" in text and "- claude-haiku-5-5" in text
    assert "preferred_models_only: true" in text


def test_all_eligible_clears_the_model_list(cfgfile):
    assert cmd_model_routing_configure(_args(models="a,b", all_eligible=False, selected_only=False,
                                             strategy=None, threshold=None, allow_unknown=None,
                                             cost_opt=None)) == 0
    assert cmd_model_routing_configure(_args(models=None, all_eligible=True, selected_only=False,
                                             strategy=None, threshold=None, allow_unknown=None,
                                             cost_opt=None)) == 0
    text = cfgfile.read_text()
    assert "- a" not in text and "models: []" in text


def test_invalid_value_writes_nothing(cfgfile, capsys):
    before = cfgfile.read_text()
    args = _args(strategy=None, threshold=5.0, models=None, all_eligible=False,
                 selected_only=False, allow_unknown=None, cost_opt=None)
    assert cmd_model_routing_configure(args) == 1
    assert cfgfile.read_text() == before       # byte-identical
    assert "nothing written" in capsys.readouterr().out


def test_cancel_writes_nothing(cfgfile, monkeypatch):
    before = cfgfile.read_text()
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert cmd_model_routing_enable(_args(yes=False)) == 0
    assert cfgfile.read_text() == before


def test_enable_then_configure_keeps_enabled(cfgfile):
    """A later option change must not silently turn routing off."""
    assert cmd_model_routing_enable(_args()) == 0
    assert cmd_model_routing_configure(_args(strategy="quality", threshold=None, models=None,
                                             all_eligible=False, selected_only=False,
                                             allow_unknown=None, cost_opt=None)) == 0
    text = cfgfile.read_text()
    assert "enabled: true" in text and "strategy: quality" in text
