"""`hermes model routing` — the CLI surface for dynamic model selection.

Behaviour contracts (not snapshots):

* ``hermes model`` with no subcommand still reaches the interactive picker, and
  ``--refresh`` keeps working — the routing verbs are additive.
* ``enable`` / ``disable`` round-trip through ``config.yaml`` and never disturb
  unrelated keys.
* ``explain`` prints a decision for a real prompt, and says plainly that there
  is nothing to route when only one model is configured, instead of inventing a
  choice.

Ported from the pre-schema-31 tree.  The setup-wizard prompt tests
(``_setup_routing_prompt``) are deliberately NOT ported: this lineage's wizard
writes ``smart_model_routing.enabled = False`` as a blank-slate default
(``hermes_cli/setup_quick.py``) and has no interactive routing prompt, and
``tests/hermes_cli/test_setup_blank_slate.py`` pins that.
"""

from __future__ import annotations

import argparse

import pytest


# ---------------------------------------------------------------------------
# parser wiring
# ---------------------------------------------------------------------------

def _build():
    from hermes_cli.subcommands.model import build_model_parser

    def cmd_model(args):  # pragma: no cover - identity only
        return "picker"

    def cmd_model_routing(args):  # pragma: no cover - identity only
        return "routing"

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_model_parser(sub, cmd_model=cmd_model, cmd_model_routing=cmd_model_routing)
    return parser, cmd_model, cmd_model_routing


@pytest.mark.parametrize(
    "argv",
    [
        ["model", "routing", "status"],
        ["model", "routing", "enable"],
        ["model", "routing", "disable"],
        ["model", "routing", "explain", "what", "is", "2+2"],
    ],
)
def test_routing_verbs_reach_the_routing_handler(argv):
    parser, _picker, routing = _build()
    ns = parser.parse_args(argv)
    assert ns.func is routing


def test_bare_model_still_reaches_the_interactive_picker():
    """Adding subcommands must not steal `hermes model` itself."""
    parser, picker, _routing = _build()
    assert parser.parse_args(["model"]).func is picker
    ns = parser.parse_args(["model", "--refresh"])
    assert ns.func is picker
    assert ns.refresh is True


def test_explain_accepts_candidate_and_modality_overrides():
    parser, _picker, _routing = _build()
    ns = parser.parse_args(
        [
            "model", "routing", "explain", "make", "it", "brighter",
            "--candidates", "a,b",
            "--attachments", "image",
            "--conversation-tokens", "1200",
        ]
    )
    assert ns.prompt == ["make", "it", "brighter"]
    assert ns.candidates == "a,b"
    assert ns.attachments == "image"
    assert ns.conversation_tokens == 1200


def test_builder_without_routing_handler_resolves_it_lazily():
    """Callers that only inject cmd_model (existing tests) must keep working."""
    from hermes_cli.subcommands.model import build_model_parser

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_model_parser(sub, cmd_model=lambda a: "picker")
    ns = parser.parse_args(["model", "routing", "status"])
    assert callable(ns.func)


def test_cli_parser_registers_the_routing_handler():
    """The real `hermes` parser must expose `model routing` (not a stub)."""
    from hermes_cli.main import _build_cli_parser

    parser, _sub = _build_cli_parser()
    ns = parser.parse_args(["model", "routing", "status"])
    from hermes_cli.main import cmd_model_routing

    assert ns.func is cmd_model_routing


# ---------------------------------------------------------------------------
# config round-trip
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


def _ns(**kw):
    base = dict(routing_command=None, prompt=None, candidates=None,
                current_model=None, provider=None, attachments=None,
                conversation_tokens=0, strategy=None, stability_threshold=None,
                switch_on_cost=None, all_models=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _routing_section(home):
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return cfg.load_config().get("smart_model_routing") or {}


def test_enable_disable_roundtrip(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="enable", strategy="cost")) == 0
    section = _routing_section(home)
    assert section.get("enabled") is True
    assert section.get("strategy") == "cost"

    assert cmd_model_routing(_ns(routing_command="disable")) == 0
    section = _routing_section(home)
    assert section.get("enabled") is False
    # disabling must not silently discard the user's other preferences
    assert section.get("strategy") == "cost"


def test_enable_all_models_flips_preferred_models_only(home):
    from hermes_cli.main import cmd_model_routing

    cmd_model_routing(_ns(routing_command="enable", all_models=True))
    assert _routing_section(home).get("preferred_models_only") is False


def test_roundtrip_preserves_unrelated_config(home):
    from hermes_cli.main import cmd_model_routing

    cmd_model_routing(_ns(routing_command="enable"))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    raw = cfg.load_config()
    assert (raw.get("display") or {}).get("skin") == "slate"
    assert (raw.get("model") or {}).get("default") == "model-a"


def test_enable_writes_a_config_validating_section(home):
    """`smart_model_routing` must stay a recognized root after the write."""
    from hermes_cli.config import load_config, validate_config_structure
    from hermes_cli.main import cmd_model_routing

    cmd_model_routing(_ns(routing_command="enable"))
    issues = validate_config_structure(load_config())
    offending = [i.message for i in issues if "smart_model_routing" in i.message]
    assert not offending, f"the written section must validate: {offending}"


# ---------------------------------------------------------------------------
# status / explain output
# ---------------------------------------------------------------------------

def test_status_reports_candidates(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status")) == 0
    out = capsys.readouterr().out
    assert "Dynamic model selection: OFF" in out
    assert "model-a" in out and "model-b" in out


def test_explain_prints_a_decision(home, capsys):
    from hermes_cli.main import cmd_model_routing

    code = cmd_model_routing(
        _ns(routing_command="explain", prompt=["What", "is", "2+2?"])
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "complexity:" in out
    assert "reason:" in out
    assert "model:" in out


def test_explain_says_nothing_to_route_with_one_model(tmp_path, monkeypatch, capsys):
    home = tmp_path / "single"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    (home / "config.yaml").write_text(
        "model:\n  default: only-model\n  provider: prov\n", encoding="utf-8"
    )

    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="explain", prompt=["hello"])) == 0
    out = capsys.readouterr().out
    assert "Nothing to route" in out
    assert "only-model" in out


def test_explain_candidates_override_config(home, capsys):
    """--candidates previews a list the config does not contain yet."""
    from hermes_cli.main import cmd_model_routing

    code = cmd_model_routing(
        _ns(
            routing_command="explain",
            prompt=["summarise", "this", "thread"],
            candidates="x-model,y-model",
        )
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "x-model" in out or "y-model" in out
