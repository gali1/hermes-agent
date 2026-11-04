"""`hermes model routing configure` — scripted flags and the interactive flow.

Contracts pinned here:

* The flag path needs no TTY and writes only through
  ``agent.model_router.update_routing`` — the same writer the interactive flow,
  the setup wizard and the gateway use.  Nothing else touches config.yaml.
* The interactive flow never writes before the final confirmation, so
  cancelling at ANY prompt leaves the config file byte-identical.
* A validation failure exits non-zero carrying the validator's own message
  (which names the offending key) and writes nothing.
* ``hermes --help`` mentions routing; ``hermes model --help`` keeps listing it.
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
        "  other:\n"
        "    models:\n"
        "      - other-a\n"
        "display:\n"
        "  skin: slate\n",
        encoding="utf-8",
    )
    return home


def _config_text(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def _reload():
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return cfg.load_config()


def _section(home) -> dict:
    return _reload().get("smart_model_routing") or {}


def _ns(**kw):
    """A Namespace shaped like the real parser's, for direct handler calls."""
    base = dict(
        routing_command="configure", prompt=None, candidates=None,
        current_model=None, provider=None, attachments=None,
        conversation_tokens=0, strategy=None, stability_threshold=None,
        switch_on_cost=None, all_models=False, models=None, enable=False,
        disable=False, cost_optimization=None, dry_run=False, json=False,
        yes=False,
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


# ---------------------------------------------------------------------------
# parser surface
# ---------------------------------------------------------------------------

def test_configure_parses_every_non_interactive_flag():
    parser = _build()
    ns = parser.parse_args(
        [
            "model", "routing", "configure",
            "--models", "model-a,model-b",
            "--models", "model-b",
            "--strategy", "quality",
            "--stability-threshold", "0.4",
            "--enable",
            "--json",
            "-y",
        ]
    )
    assert ns.routing_command == "configure"
    assert ns.models == ["model-a,model-b", "model-b"]  # split by the handler
    assert ns.strategy == "quality"
    assert ns.stability_threshold == pytest.approx(0.4)
    assert ns.enable is True
    assert ns.json is True
    assert ns.yes is True


def test_configure_accepts_yes_long_form_and_modes():
    parser = _build()
    ns = parser.parse_args(
        ["model", "routing", "configure", "--all-models", "--disable",
         "--no-cost-optimization", "--yes"]
    )
    assert ns.all_models is True
    assert ns.disable is True
    assert ns.cost_optimization is False
    assert ns.yes is True


def test_configure_enable_and_disable_are_mutually_exclusive():
    parser = _build()
    with pytest.raises(SystemExit):
        parser.parse_args(["model", "routing", "configure", "--enable", "--disable"])


def test_configure_reaches_the_routing_handler():
    parser = _build()
    assert parser.parse_args(["model", "routing", "configure"]).func is not None
    from hermes_cli.main import _build_cli_parser, cmd_model_routing

    real, _sub = _build_cli_parser()
    assert real.parse_args(["model", "routing", "configure"]).func is cmd_model_routing


# ---------------------------------------------------------------------------
# root-help discoverability (M0 gap: `hermes --help` said nothing about routing)
# ---------------------------------------------------------------------------

def test_root_help_mentions_routing():
    from hermes_cli.main import _build_cli_parser

    parser, _sub = _build_cli_parser()
    help_text = parser.format_help().lower()
    # The row wraps, so the mention can sit on the continuation line.
    assert "dynamic model routing" in help_text
    model_row = next(
        line for line in help_text.splitlines() if line.strip().startswith("model ")
    )
    assert "model" in model_row


def test_model_help_still_lists_routing():
    from hermes_cli.main import _build_cli_parser

    parser, sub = _build_cli_parser()
    # `hermes model --help` renders the model subparser; `routing` must stay a row.
    model_parser = sub.choices["model"]
    model_sub = next(a for a in model_parser._actions if isinstance(a, argparse._SubParsersAction))
    assert "routing" in model_sub.choices
    routing_parser = model_sub.choices["routing"]
    routing_sub = next(
        a for a in routing_parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert {"status", "enable", "disable", "explain", "configure"} <= set(routing_sub.choices)
    assert "routing" in model_parser.format_help()


# ---------------------------------------------------------------------------
# non-interactive: models / mode / preferences
# ---------------------------------------------------------------------------

def test_flag_path_sets_models_mode_and_preferences(home, capsys):
    from hermes_cli.main import cmd_model_routing

    code = cmd_model_routing(
        _ns(models=["model-b,model-a"], strategy="cost",
            stability_threshold=0.2, enable=True, yes=True)
    )
    assert code == 0
    section = _section(home)
    assert section["enabled"] is True
    assert section["models"] == ["model-b", "model-a"]  # order = priority
    assert section["preferred_models_only"] is True
    assert section["strategy"] == "cost"
    assert section["stability_threshold"] == pytest.approx(0.2)

    out = capsys.readouterr().out
    assert "Saved to config.yaml" in out
    assert "Verify with: hermes model routing status" in out
    # unrelated config is untouched
    reloaded = _reload()
    assert (reloaded.get("display") or {}).get("skin") == "slate"
    assert (reloaded.get("providers") or {}).get("other", {}).get("models") == ["other-a"]


def test_flag_path_all_models_clears_the_selection(home):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(models=["model-a"], yes=True)) == 0
    assert _section(home)["models"] == ["model-a"]
    assert _section(home)["preferred_models_only"] is True

    assert cmd_model_routing(_ns(all_models=True, yes=True)) == 0
    section = _section(home)
    assert section["models"] == []
    assert section["preferred_models_only"] is False


def test_flag_path_rejects_a_model_the_provider_does_not_list(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(models=["not-a-real-model"], yes=True)) == 2
    err = capsys.readouterr().err
    assert "not-a-real-model" in err
    assert "model-a, model-b" in err  # the known list, so the fix is obvious
    assert _config_text(home) == before  # nothing written


def test_flag_path_rejects_models_and_all_models_together(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(models=["model-a"], all_models=True)) == 2
    assert "only one" in capsys.readouterr().err
    assert _config_text(home) == before


def test_flag_path_without_changes_is_read_only(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    out = capsys.readouterr().out
    assert "Dynamic model selection: OFF" in out
    assert "Nothing to change" in out
    assert _config_text(home) == before


def test_flag_path_dry_run_writes_nothing(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(enable=True, models=["model-b"], dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "nothing written" in out
    assert _config_text(home) == before


def test_flag_path_json_reports_settings_read_back_from_disk(home, capsys):
    import json

    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(
        _ns(enable=True, models=["model-b"], strategy="quality", json=True)
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["written"] is True
    assert payload["dry_run"] is False
    assert set(payload["changed"]) == {"enabled", "models", "preferred_models_only", "strategy"}
    # read back from disk, not from the object we asked to write
    assert payload["settings"]["enabled"] is True
    assert payload["settings"]["models"] == ["model-b"]
    assert payload["settings"]["strategy"] == "quality"
    assert payload["path"].endswith("config.yaml")


def test_flag_path_validation_failure_exits_non_zero_with_the_validator_message(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(stability_threshold=2.0, yes=True)) == 1
    err = capsys.readouterr().err
    assert "smart_model_routing.stability_threshold" in err  # names the key
    assert "between 0 and 1" in err
    assert _config_text(home) == before  # validated before any I/O


def test_configure_json_without_changes_is_a_read_only_report(home, capsys):
    import json

    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] is False
    assert payload["changed"] == []
    assert payload["settings"]["enabled"] is False
    assert _config_text(home) == before


# ---------------------------------------------------------------------------
# interactive flow
# ---------------------------------------------------------------------------

class _Script:
    """Scripted answers for the interactive flow's prompt wrappers."""

    def __init__(self, choices=(), texts=(), models=(), confirm=True):
        self._choices = list(choices)
        self._texts = list(texts)
        self._models = list(models)
        self._confirm = confirm
        self.asked: list[str] = []

    def choice(self, question, choices, default=0):
        self.asked.append(question)
        return self._choices.pop(0)

    def text(self, question, default=""):
        self.asked.append(question)
        return self._texts.pop(0)

    def pick(self, models, current=""):
        self.asked.append("pick-model")
        return self._models.pop(0)

    def confirm(self, question, *, default=False):
        self.asked.append(question)
        return self._confirm


@pytest.fixture
def interactive(monkeypatch):
    """Force the TTY branch and route every prompt through a script."""
    import hermes_cli.main as main

    monkeypatch.setattr(main, "_routing_is_tty", lambda: True)
    scripts: dict = {}

    def _install(script):
        scripts["s"] = script
        monkeypatch.setattr(main, "_routing_ask_choice", script.choice)
        monkeypatch.setattr(main, "_routing_ask_text", script.text)
        monkeypatch.setattr(main, "_routing_pick_model", script.pick)
        monkeypatch.setattr(main, "_routing_confirm", script.confirm)
        return script

    _install.scripts = scripts
    return _install


def test_interactive_enable_select_models_and_save(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    script = interactive(
        _Script(
            choices=[1, 1, 0, 1, 1, 0, 0],  # enable, selected-only, add-another, done, cost, cheaper, advanced-keep
            texts=["0.25"],
            models=["model-a", "model-b"],
            confirm=True,
        )
    )
    assert cmd_model_routing(_ns()) == 0

    section = _section(home)
    assert section["enabled"] is True
    assert section["models"] == ["model-a", "model-b"]
    assert section["preferred_models_only"] is True
    assert section["strategy"] == "cost"
    assert section["stability_threshold"] == pytest.approx(0.25)
    assert section["cost_optimization"] is True

    out = capsys.readouterr().out
    assert "── Summary ──" in out
    assert "before:" in out and "after:" in out
    assert "would select:" in out          # routing_preview() output
    assert "tier:" in out
    assert "Verify with: hermes model routing status" in out
    assert any("Save these settings?" in q for q in script.asked)


def test_interactive_cancel_at_the_first_prompt_changes_nothing(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    interactive(_Script(choices=[None]))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    assert "Cancelled" in capsys.readouterr().out
    assert _config_text(home) == before


def test_interactive_cancel_at_the_model_picker_changes_nothing(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    # enable, only-selected, then the picker is cancelled
    interactive(_Script(choices=[1, 1], models=[None]))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    assert "Cancelled" in capsys.readouterr().out
    assert _config_text(home) == before


def test_interactive_cancel_at_a_preference_changes_nothing(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    # enable, all-models, strategy, then cancel at the cost prompt
    interactive(_Script(choices=[1, 0, 0, None], texts=[""]))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    assert "Cancelled" in capsys.readouterr().out
    assert _config_text(home) == before


def test_interactive_declining_the_confirmation_writes_nothing(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    interactive(_Script(choices=[1, 0, 0, 0, 0], texts=[""], confirm=False))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    out = capsys.readouterr().out
    assert "Cancelled" in out
    assert _config_text(home) == before


def test_interactive_yes_skips_the_confirmation(home, capsys, interactive, monkeypatch):
    from hermes_cli.main import cmd_model_routing

    script = interactive(_Script(choices=[1, 0, 0, 0, 0], texts=[""], confirm=False))
    assert cmd_model_routing(_ns(yes=True)) == 0
    assert not any("Save these settings?" in q for q in script.asked)
    assert _section(home)["enabled"] is True


def test_interactive_bad_threshold_uses_the_validator_message(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    interactive(_Script(choices=[1, 0, 0, 0, 0], texts=["2.5"], confirm=True))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 1
    err = capsys.readouterr().err
    assert "smart_model_routing.stability_threshold" in err
    assert "between 0 and 1" in err
    assert _config_text(home) == before


def test_interactive_no_change_is_reported_without_a_write(home, capsys, interactive):
    """A full pass that changes nothing must not rewrite config.yaml."""
    from hermes_cli.main import cmd_model_routing

    (home / "config.yaml").write_text(
        "model:\n"
        "  default: model-a\n"
        "  provider: prov\n"
        "providers:\n"
        "  prov:\n"
        "    models:\n"
        "      - model-a\n"
        "      - model-b\n"
        "smart_model_routing:\n"
        "  enabled: true\n"
        "  models: []\n"
        "  preferred_models_only: false\n"
        "  strategy: automatic\n"
        "  cost_optimization: true\n"
        "  stability_threshold: 0.15\n",
        encoding="utf-8",
    )
    _reload()

    # keep it on, all eligible models, automatic, keep the threshold, cheaper
    interactive(_Script(choices=[0, 0, 0, 0, 0], texts=[""], confirm=True))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    assert "No change" in capsys.readouterr().out
    assert _config_text(home) == before


def test_interactive_all_eligible_mode_clears_the_selection(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(models=["model-a"], yes=True)) == 0
    assert _section(home)["preferred_models_only"] is True

    # keep/off choice 2 = disable? no: 0 keeps, 0 = all eligible, automatic, keep, cheaper
    interactive(_Script(choices=[0, 0, 0, 0, 0], texts=["0.15"], confirm=True))
    assert cmd_model_routing(_ns()) == 0
    section = _section(home)
    assert section["models"] == []
    assert section["preferred_models_only"] is False
