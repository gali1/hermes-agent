"""`hermes model routing configure` — the advanced rules the engine actually reads.

Directive §4 requires every routing setting the runtime reads to be reachable
from the interfaces, and forbids controls that map to nothing.  Before this
module existed, ``exclude``, ``allow_unknown_metadata``, ``switch_on_cost``,
``weights`` and ``reliability`` were read by ``agent/model_router.py`` but could
only be set by hand-editing ``smart_model_routing``.

Contracts pinned here:

* Every one of them is settable through ``configure`` — both the flag path and
  the interactive flow — and every write goes through
  ``agent.model_router.update_routing`` (no second writer, no second schema).
* A value the validator rejects fails with the validator's own message (which
  names the key) and leaves ``config.yaml`` byte-identical.
* Repeating the same command is a no-op: the weights merge base is the
  *authored* mapping, not the normalised effective one.  (Merging over the
  normalised values re-scales every number on each run, so the same command
  would rewrite the file forever.)
* Cancelling at the advanced step, or any later prompt, writes nothing.
* What the CLI writes is what the engine reads: the same
  ``routing_settings()`` values drive eligibility and scoring.
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
        "      - model-c\n"
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
    base = dict(
        routing_command="configure", prompt=None, candidates=None,
        current_model=None, provider=None, attachments=None,
        conversation_tokens=0, strategy=None, stability_threshold=None,
        switch_on_cost=None, all_models=False, models=None, enable=False,
        disable=False, cost_optimization=None, dry_run=False, json=False,
        yes=False, exclude=None, clear_exclude=False,
        allow_unknown_metadata=None, weights=None, weights_default=False,
        reliability=None, clear_reliability=False,
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

ADVANCED_FLAGS = (
    "--exclude", "--clear-exclude",
    "--allow-unknown-metadata", "--no-allow-unknown-metadata",
    "--switch-on-cost", "--no-switch-on-cost",
    "--weights", "--weights-default",
    "--reliability", "--clear-reliability",
)


def test_configure_parser_accepts_every_advanced_flag():
    """One parse of all of them at once — the surface a script uses."""
    parser = _build()
    ns = parser.parse_args(
        [
            "model", "routing", "configure",
            "--exclude", "model-b,model-c",
            "--weights", "capability=0.5,cost=0.3",
            "--reliability", "model-a=0.97",
            "--no-allow-unknown-metadata",
            "--switch-on-cost",
            "--yes",
        ]
    )
    assert ns.exclude == ["model-b,model-c"]      # split by the handler
    assert ns.weights == ["capability=0.5,cost=0.3"]
    assert ns.reliability == ["model-a=0.97"]
    assert ns.allow_unknown_metadata is False
    assert ns.switch_on_cost is True
    assert ns.yes is True


def test_configure_help_text_documents_them():
    from hermes_cli.main import _build_cli_parser

    _parser, sub = _build_cli_parser()
    model_parser = sub.choices["model"]
    model_sub = next(
        a for a in model_parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert "routing" in model_parser.format_help()
    routing_parser = model_sub.choices["routing"]
    routing_sub = next(
        a for a in routing_parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    configure_help = routing_sub.choices["configure"].format_help()
    for flag in ADVANCED_FLAGS:
        assert flag in configure_help, flag


# ---------------------------------------------------------------------------
# flag path
# ---------------------------------------------------------------------------

def test_flags_write_every_advanced_setting(home, capsys):
    from hermes_cli.main import cmd_model_routing

    code = cmd_model_routing(
        _ns(
            exclude=["model-b, model-c"],
            allow_unknown_metadata=False,
            switch_on_cost=True,
            weights=["capability=0.6,cost=0.2"],
            reliability=["model-a=0.97,model-b=0.9"],
            yes=True,
        )
    )
    assert code == 0
    section = _section(home)
    assert section["exclude"] == ["model-b", "model-c"]
    assert section["allow_unknown_metadata"] is False
    assert section["switch_on_cost"] is True
    assert section["weights"]["capability"] == pytest.approx(0.6)
    assert section["weights"]["cost"] == pytest.approx(0.2)
    # unnamed dimensions keep their default value instead of disappearing
    assert section["weights"]["continuity"] == pytest.approx(0.05)
    assert section["reliability"] == {
        "model-a": pytest.approx(0.97),
        "model-b": pytest.approx(0.9),
    }

    out = capsys.readouterr().out
    assert "never auto-select:   model-b, model-c" in out
    assert "strict" in out
    # unrelated config survives
    reloaded = _reload()
    assert (reloaded.get("display") or {}).get("skin") == "slate"
    assert (reloaded.get("providers") or {}).get("prov", {}).get("models") == [
        "model-a", "model-b", "model-c",
    ]


def test_repeating_the_same_flags_is_a_no_op(home, capsys):
    """Idempotency: merging weights over the authored mapping, not the normalised one."""
    from hermes_cli.main import cmd_model_routing

    args = dict(
        exclude=["model-b"], weights=["capability=0.6,cost=0.2"],
        reliability=["model-a=0.97"], switch_on_cost=True, yes=True,
    )
    assert cmd_model_routing(_ns(**args)) == 0
    first = _config_text(home)
    capsys.readouterr()

    assert cmd_model_routing(_ns(**args)) == 0
    out = capsys.readouterr().out
    assert "No change" in out
    assert _config_text(home) == first


def test_unknown_weight_dimension_is_rejected_by_the_validator(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(weights=["bird=0.4"], yes=True)) == 1
    err = capsys.readouterr().err
    assert "smart_model_routing.weights.bird" in err
    assert "not a scoring dimension" in err
    assert _config_text(home) == before


def test_weight_syntax_error_is_rejected_before_any_write(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(weights=["capability"], yes=True)) == 2
    assert "must look like name=value" in capsys.readouterr().err
    assert _config_text(home) == before


def test_out_of_range_reliability_is_rejected(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(reliability=["model-a=1.4"], yes=True)) == 1
    err = capsys.readouterr().err
    assert "smart_model_routing.reliability.model-a" in err
    assert "between 0 and 1" in err
    assert _config_text(home) == before


def test_weights_default_resets_to_the_engine_defaults(home, capsys):
    from agent import model_router as mr
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(weights=["capability=0.9,cost=0.05"], yes=True)) == 0
    assert _section(home)["weights"]["capability"] == pytest.approx(0.9)
    capsys.readouterr()

    assert cmd_model_routing(_ns(weights_default=True, yes=True)) == 0
    section = _section(home)
    assert section["weights"] == pytest.approx(mr.default_weights())
    assert "changed: weights" in capsys.readouterr().out


def test_clear_flags_empty_the_lists(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(
        _ns(exclude=["model-b"], reliability=["model-a=0.9"], yes=True)) == 0
    assert _section(home)["exclude"] == ["model-b"]
    assert _section(home)["reliability"] == {"model-a": pytest.approx(0.9)}
    capsys.readouterr()

    assert cmd_model_routing(_ns(clear_exclude=True, clear_reliability=True, yes=True)) == 0
    section = _section(home)
    assert section["exclude"] == []
    assert section["reliability"] == {}


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        (dict(exclude=["model-b"], clear_exclude=True), "--clear-exclude"),
        (dict(weights=["cost=0.2"], weights_default=True), "--weights-default"),
        (dict(reliability=["model-a=0.9"], clear_reliability=True), "--clear-reliability"),
    ],
)
def test_conflicting_flags_are_refused(home, capsys, kwargs, needle):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(yes=True, **kwargs)) == 2
    assert needle in capsys.readouterr().err
    assert _config_text(home) == before


def test_dry_run_reports_without_writing(home, capsys):
    from hermes_cli.main import cmd_model_routing

    before = _config_text(home)
    assert cmd_model_routing(_ns(exclude=["model-c"], dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "nothing written" in out
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
    import hermes_cli.main as main

    monkeypatch.setattr(main, "_routing_is_tty", lambda: True)

    def _install(script):
        monkeypatch.setattr(main, "_routing_ask_choice", script.choice)
        monkeypatch.setattr(main, "_routing_ask_text", script.text)
        monkeypatch.setattr(main, "_routing_pick_model", script.pick)
        monkeypatch.setattr(main, "_routing_confirm", script.confirm)
        return script

    return _install


def test_interactive_sets_every_advanced_setting(home, capsys, interactive):
    """One menu: exclude → downgrade → unknown metadata → weights → reliability."""
    from hermes_cli.main import cmd_model_routing

    script = interactive(
        _Script(
            choices=[
                1,   # enable
                0,   # all eligible models
                0,   # strategy: automatic
                0,   # cost: prefer cheaper
                3,   # advanced: never auto-select
                2,   # advanced: mid-session cost downgrade -> on
                1,   # advanced: allow unknown metadata -> off (strict)
                4,   # advanced: weights
                5,   # advanced: reliability
                0,   # advanced: done
            ],
            texts=["", "model-b, model-c", "capability=0.6,cost=0.2", "model-a=0.93"],
            confirm=True,
        )
    )
    assert cmd_model_routing(_ns()) == 0

    section = _section(home)
    assert section["enabled"] is True
    assert section["exclude"] == ["model-b", "model-c"]
    assert section["switch_on_cost"] is True
    assert section["allow_unknown_metadata"] is False
    assert section["weights"]["capability"] == pytest.approx(0.6)
    assert section["reliability"] == {"model-a": pytest.approx(0.93)}

    out = capsys.readouterr().out
    assert any("Advanced selection rules:" in q for q in script.asked)
    assert "Save these settings?" in " ".join(script.asked)
    assert "── Summary ──" in out


def test_interactive_advanced_done_keeps_the_current_values(home, capsys, interactive):
    """Pressing "keep" must not write a weights block the file did not have."""
    from hermes_cli.main import cmd_model_routing

    (home / "config.yaml").write_text(
        _config_text(home)
        + "smart_model_routing:\n"
          "  enabled: true\n"
          "  models: []\n"
          "  preferred_models_only: false\n"
          "  strategy: automatic\n"
          "  cost_optimization: true\n"
          "  stability_threshold: 0.15\n",
        encoding="utf-8",
    )
    _reload()
    before = _config_text(home)

    interactive(_Script(choices=[0, 0, 0, 0, 0], texts=[""], confirm=True))
    assert cmd_model_routing(_ns()) == 0
    assert "No change" in capsys.readouterr().out
    assert _config_text(home) == before


def test_interactive_cancel_at_the_advanced_step_changes_nothing(home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    # enable, all eligible, automatic, keep threshold, cheaper, then cancel
    interactive(_Script(choices=[1, 0, 0, 0, None], texts=[""]))
    before = _config_text(home)
    assert cmd_model_routing(_ns()) == 0
    assert "Cancelled" in capsys.readouterr().out
    assert _config_text(home) == before


def test_interactive_bad_weight_value_reprompts_and_keeps_the_pending_state(
        home, capsys, interactive):
    from hermes_cli.main import cmd_model_routing

    # advanced: weights -> "capability" (no '='), then done, then save
    interactive(
        _Script(
            choices=[1, 0, 0, 0, 4, 0],
            texts=["", "capability"],
            confirm=True,
        )
    )
    assert cmd_model_routing(_ns()) == 0
    err = capsys.readouterr().err
    assert "must look like name=value" in err
    # The flow continued and saved the rest; the bad value was never written.
    assert "weights" not in _section(home)


# ---------------------------------------------------------------------------
# the control maps to something the engine reads
# ---------------------------------------------------------------------------

def test_the_engine_reads_what_the_cli_writes(home, capsys):
    from agent import model_router as mr
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(
        _ns(exclude=["model-b"], weights=["capability=0.7,cost=0.1"],
            reliability=["model-a=0.8"], allow_unknown_metadata=False, yes=True)
    ) == 0

    settings = mr.routing_settings()          # reads config.yaml itself: the CLI view
    assert settings["exclude"] == ["model-b"]
    assert settings["reliability"] == {"model-a": pytest.approx(0.8)}
    assert settings["allow_unknown_metadata"] is False
    # 0.7 / (0.7 + 0.1 + defaults 0.1 + 0.1 + 0.05) — the engine normalises weights.
    assert settings["weights"]["capability"] == pytest.approx(0.7 / 1.05, abs=1e-4)

    # … and the excluded model is removed before ranking, with a named reason.
    preview = mr.routing_preview(
        "summarise this", _reload(),
        candidates=["model-a", "model-b", "model-c"], current_model="model-a",
    )
    reasons = {row["model"]: row["excluded_reason"] for row in preview.get("candidates") or []}
    assert reasons.get("model-b") == "excluded"


def test_status_shows_a_non_default_rule(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(exclude=["model-c"], allow_unknown_metadata=False, yes=True)) == 0
    capsys.readouterr()
    assert cmd_model_routing(_ns()) == 0
    out = capsys.readouterr().out
    assert "never auto-select:        model-c" in out
    assert "unknown metadata:         strict" in out


def test_status_keeps_the_default_view_clean(home, capsys):
    """A default block must not grow rows that describe nothing."""
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns()) == 0
    out = capsys.readouterr().out
    assert "never auto-select" not in out
    assert "unknown metadata" not in out
