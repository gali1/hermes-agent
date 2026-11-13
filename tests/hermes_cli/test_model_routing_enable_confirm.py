"""`hermes model routing enable` — the confirmation gate on the one action that
changes behaviour on every later turn.

Contracts pinned here (directive §3.7 and §5.1):

* Enabling routing from a terminal shows what it would work from and asks for an
  explicit yes first; declining writes nothing and leaves the file
  byte-identical.
* ``--yes`` is the assume-yes escape hatch, the same flag shape ``configure``
  uses.
* A caller with no terminal (script, cron, gateway) is never prompted — a
  prompt there would hang instead of doing what was asked — so it keeps the
  established non-interactive behaviour.
* ``disable`` never prompts: turning routing off restores the single-model
  behaviour, so there is nothing to warn about.
* Asking to enable something already enabled is not a state change and does not
  prompt either.

Nothing here touches a real config: ``HERMES_HOME`` points at ``tmp_path``.
"""

from __future__ import annotations

import argparse

import pytest


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: model-a\n"
        "  provider: prov\n"
        "providers:\n"
        "  prov:\n"
        "    models:\n"
        "      - model-a\n"
        "      - model-b\n"
        "display:\n"
        "  skin: slate\n",
        encoding="utf-8",
    )
    return home


def _text(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def _section(home) -> dict:
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return cfg.load_config().get("smart_model_routing") or {}


def _ns(**kw):
    base = dict(
        routing_command="enable", prompt=None, candidates=None, current_model=None,
        provider=None, attachments=None, conversation_tokens=0, strategy=None,
        stability_threshold=None, switch_on_cost=None, all_models=False,
        models=None, enable=False, disable=False, cost_optimization=None,
        dry_run=False, json=False, yes=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _build():
    from hermes_cli.subcommands.model import build_model_parser

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_model_parser(
        sub, cmd_model=lambda a: "picker", cmd_model_routing=lambda a: "routing"
    )
    return parser


def _no_prompt_allowed(monkeypatch):
    """A prompt at this point is the failure the test is looking for."""

    def _boom(*_a, **_kw):  # pragma: no cover - only runs on failure
        raise AssertionError("the confirmation prompt must not be reached here")

    monkeypatch.setattr("hermes_cli.main._routing_confirm", _boom)


def _answer(monkeypatch, value, seen=None):
    def _confirm(question, *, default=False):
        if seen is not None:
            seen.append((question, default))
        return value

    monkeypatch.setattr("hermes_cli.main._routing_confirm", _confirm)


def _tty(monkeypatch, value):
    monkeypatch.setattr("hermes_cli.main._routing_is_tty", lambda: bool(value))


# ---------------------------------------------------------------------------
# parser surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["-y", "--yes"])
def test_enable_accepts_the_assume_yes_flag(flag):
    args = _build().parse_args(["model", "routing", "enable", flag])
    assert args.yes is True
    assert args.routing_command == "enable"


def test_enable_without_the_flag_defaults_to_asking():
    args = _build().parse_args(["model", "routing", "enable"])
    assert args.yes is False


def test_enable_still_parses_its_settings_flags():
    args = _build().parse_args(
        ["model", "routing", "enable", "--strategy", "cost",
         "--stability-threshold", "0.3", "--all-models"]
    )
    assert (args.strategy, args.stability_threshold, args.all_models) == ("cost", 0.3, True)


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------

def test_enable_on_a_terminal_shows_the_preview_then_writes(home, capsys, monkeypatch):
    from hermes_cli import main

    _tty(monkeypatch, True)
    seen = []
    _answer(monkeypatch, True, seen)

    rc = main.cmd_model_routing(_ns())

    assert rc == 0
    out = capsys.readouterr().out
    assert "would work from" in out, "the preview must be shown before the question"
    assert "candidates (2)" in out and "model-a" in out, "the preview names the pool"
    assert seen and seen[0][0] == "Enable dynamic model selection?"
    assert seen[0][1] is False, "the safe answer must be the default"
    assert _section(home).get("enabled") is True


def test_enable_declined_on_a_terminal_leaves_the_config_byte_identical(home, capsys, monkeypatch):
    from hermes_cli import main

    _tty(monkeypatch, True)
    _answer(monkeypatch, False)
    before = _text(home)

    rc = main.cmd_model_routing(_ns())

    assert rc == 0
    assert _text(home) == before, "declining must not write"
    assert "smart_model_routing" not in _section(home)
    assert "Cancelled" in capsys.readouterr().out


def test_enable_with_yes_does_not_prompt_on_a_terminal(home, monkeypatch):
    from hermes_cli import main

    _tty(monkeypatch, True)
    _no_prompt_allowed(monkeypatch)

    assert main.cmd_model_routing(_ns(yes=True)) == 0
    assert _section(home).get("enabled") is True


def test_enable_without_a_terminal_never_prompts(home, monkeypatch):
    """Scripts, cron and the gateway cannot answer — they must not be asked."""
    from hermes_cli import main

    _tty(monkeypatch, False)
    _no_prompt_allowed(monkeypatch)

    assert main.cmd_model_routing(_ns()) == 0
    assert _section(home).get("enabled") is True


def test_enable_already_enabled_does_not_prompt(home, monkeypatch):
    from hermes_cli import main

    _tty(monkeypatch, True)
    assert main.cmd_model_routing(_ns(yes=True)) == 0  # switch it on first
    _no_prompt_allowed(monkeypatch)

    assert main.cmd_model_routing(_ns()) == 0
    assert _section(home).get("enabled") is True


def test_disable_never_prompts_even_on_a_terminal(home, monkeypatch):
    from hermes_cli import main

    _tty(monkeypatch, True)
    assert main.cmd_model_routing(_ns(yes=True)) == 0
    _no_prompt_allowed(monkeypatch)

    assert main.cmd_model_routing(_ns(routing_command="disable")) == 0
    assert _section(home).get("enabled") is False


def test_the_gate_shows_the_state_it_is_leaving(home, capsys, monkeypatch):
    """The preview must describe the settings being enabled, truthfully as OFF."""
    from hermes_cli import main

    _tty(monkeypatch, True)
    _answer(monkeypatch, False)

    main.cmd_model_routing(_ns())

    out = capsys.readouterr().out
    assert "Dynamic model selection: OFF" in out, "the pre-change state is reported as such"
    assert "strategy:                 automatic" in out
    assert "Saved in: config.yaml" not in out, "nothing has been saved when the question is asked"
