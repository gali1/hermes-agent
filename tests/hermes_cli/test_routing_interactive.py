"""Tests for the interactive routing flow and the multi-select prompt.

The prompts themselves are patched: what matters is that each answer maps to the right
config change, that cancel applies nothing, and that a non-TTY never blocks waiting for input.
"""

from __future__ import annotations

import argparse

import pytest

import hermes_cli.setup as setup_mod
from hermes_cli.subcommands.model import _routing_interactive


@pytest.fixture
def cfgfile(tmp_path, monkeypatch):
    import hermes_cli.config as config_mod

    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: deepseek-v4.1-flash\n  provider: opencode-zen\n")
    monkeypatch.setattr(config_mod, "get_config_path", lambda: path)
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: False)
    return path


def _answers(monkeypatch, *, choice, mode=None, models=None, yes=False, text="", strategy=0):
    """Answer each prompt by question text, so one stub cannot answer the wrong step."""

    def _choice(q, c, default=0, description=None):
        if "change" in q:
            return choice
        if "models" in q:
            return mode or 0
        return strategy

    monkeypatch.setattr(setup_mod, "prompt_choice", _choice)
    monkeypatch.setattr(setup_mod, "prompt_multi_choice", lambda *a, **k: models)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda q, default=False: yes)
    monkeypatch.setattr("builtins.input", lambda *_: text)


def _catalog(monkeypatch, names=("alpha", "beta")):
    """Supply a deterministic catalog so model selection does not depend on the live cache.

    Real ModelOption instances: explain_routing() builds candidates through the same function,
    so a duck-typed stub would break ranking with an AttributeError.
    """
    import agent.model_routing as routing_mod

    monkeypatch.setattr(
        routing_mod, "options_from_catalog",
        lambda provider, models=None: [
            routing_mod.ModelOption(name=n, provider=provider or "opencode-zen") for n in names])


# --- multi-select helper ---------------------------------------------------------------

def test_multi_choice_parses_numbers(monkeypatch):
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: "1,3")
    assert setup_mod.prompt_multi_choice("pick", ["a", "b", "c"]) == [0, 2]


@pytest.mark.parametrize("answer,expected", [("all", [0, 1, 2]), ("none", []), ("", [1])])
def test_multi_choice_keywords(monkeypatch, answer, expected):
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    assert setup_mod.prompt_multi_choice("pick", ["a", "b", "c"], preselected=[1]) == expected


def test_multi_choice_cancel_returns_none(monkeypatch):
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: "cancel")
    assert setup_mod.prompt_multi_choice("pick", ["a", "b"]) is None


def test_multi_choice_retries_on_bad_input(monkeypatch):
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: False)
    answers = iter(["9", "2"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert setup_mod.prompt_multi_choice("pick", ["a", "b"]) == [1]


def test_multi_choice_noninteractive_keeps_preselection(monkeypatch):
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: True)
    assert setup_mod.prompt_multi_choice("pick", ["a", "b"], preselected=[0]) == [0]


# --- interactive flow ------------------------------------------------------------------

def test_cancel_applies_nothing(cfgfile, monkeypatch):
    before = cfgfile.read_text()
    _answers(monkeypatch, choice=4)                      # Cancel
    assert _routing_interactive() == 0
    assert cfgfile.read_text() == before


def test_declining_the_summary_applies_nothing(cfgfile, monkeypatch):
    before = cfgfile.read_text()
    _answers(monkeypatch, choice=0, yes=False)           # Enable, then "no" at the summary
    assert _routing_interactive() == 0
    assert cfgfile.read_text() == before


def test_enable_writes(cfgfile, monkeypatch):
    _answers(monkeypatch, choice=0, yes=True)
    assert _routing_interactive() == 0
    assert "enabled: true" in cfgfile.read_text()


def test_disable_writes(cfgfile, monkeypatch):
    _answers(monkeypatch, choice=0, yes=True)
    _routing_interactive()
    _answers(monkeypatch, choice=1, yes=True)
    assert _routing_interactive() == 0
    assert "enabled: false" in cfgfile.read_text()


def test_selecting_models_sets_preferred_only(cfgfile, monkeypatch):
    """Mode 1 (only selected) must set preferred_models_only and store the picks."""
    _catalog(monkeypatch)
    _answers(monkeypatch, choice=2, mode=1, models=[0, 1], yes=True)
    assert _routing_interactive() == 0
    text = cfgfile.read_text()
    assert "preferred_models_only: true" in text
    assert "models:" in text


def test_all_eligible_clears_the_list(cfgfile, monkeypatch):
    _answers(monkeypatch, choice=2, mode=0, yes=True)
    assert _routing_interactive() == 0
    assert "models: []" in cfgfile.read_text()


def test_cancelling_model_selection_applies_nothing(cfgfile, monkeypatch):
    before = cfgfile.read_text()
    _catalog(monkeypatch)
    _answers(monkeypatch, choice=2, mode=1, models=None, yes=True)
    assert _routing_interactive() == 0
    assert cfgfile.read_text() == before


def test_preferences_strategy_and_threshold(cfgfile, monkeypatch):
    _answers(monkeypatch, choice=3, yes=True, text="0.25", strategy=1)
    assert _routing_interactive() == 0
    text = cfgfile.read_text()
    assert "strategy: cost" in text and "stability_threshold: 0.25" in text


def test_bad_threshold_applies_nothing(cfgfile, monkeypatch):
    before = cfgfile.read_text()
    _answers(monkeypatch, choice=3, yes=True, text="abc", strategy=0)
    assert _routing_interactive() == 1
    assert cfgfile.read_text() == before


def test_no_provider_refuses(cfgfile, monkeypatch):
    cfgfile.write_text("model:\n  default: x\n")
    _answers(monkeypatch, choice=0, yes=True)
    assert _routing_interactive() == 1



class _TtyStdin:
    def isatty(self):
        return True


def _pretend_tty(monkeypatch):
    """The wizard step only prompts for a real terminal; tests must opt in explicitly."""
    import sys
    monkeypatch.setattr(sys, "stdin", _TtyStdin())


# --- setup wizard hook ----------------------------------------------------------------

def test_wizard_skips_without_a_human(monkeypatch):
    """CI / dashboard-spawned actions must never block on the routing prompt."""
    from hermes_cli.model_setup_flows import _prompt_routing_after_model

    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: True)
    called = []
    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: called.append(1) or 0)
    _prompt_routing_after_model("opencode-zen")
    assert called == []          # never prompted


def test_wizard_keep_writes_nothing(cfgfile, monkeypatch):
    from hermes_cli.model_setup_flows import _prompt_routing_after_model

    _pretend_tty(monkeypatch)
    before = cfgfile.read_text()
    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: 0)
    _prompt_routing_after_model("opencode-zen")
    assert cfgfile.read_text() == before


def test_wizard_configure_delegates_to_the_shared_flow(cfgfile, monkeypatch):
    from hermes_cli.model_setup_flows import _prompt_routing_after_model

    _pretend_tty(monkeypatch)
    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: 1)
    seen = []
    import hermes_cli.subcommands.model as model_cmd
    monkeypatch.setattr(model_cmd, "_routing_interactive", lambda: seen.append(1) or 0)
    _prompt_routing_after_model("opencode-zen")
    assert seen == [1]


def test_wizard_never_raises_when_config_is_broken(monkeypatch):
    """The optional step must not break setup."""
    from hermes_cli.model_setup_flows import _prompt_routing_after_model
    import hermes_cli.config as config_mod

    _pretend_tty(monkeypatch)
    monkeypatch.setattr(setup_mod, "is_noninteractive", lambda: False)
    monkeypatch.setattr(config_mod, "read_user_config_raw",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _prompt_routing_after_model("opencode-zen")   # must not raise
