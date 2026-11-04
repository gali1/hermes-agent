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
        ["model", "routing", "configure"],
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
# `configure` — the scripted (no-TTY) path shares this module's writer
# ---------------------------------------------------------------------------
#
# The deeper configure contracts live in test_model_routing_configure.py; what
# is pinned here is that the new verb lands in the SAME section, is visible to
# the existing status renderers, and does not widen the older verbs' surface.

def test_configure_flag_path_lands_in_the_status_contract(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(
        _ns(
            routing_command="configure",
            models=["model-b"],
            strategy="quality",
            enable=True,
            yes=True,
        )
    ) == 0
    assert _routing_section(home).get("models") == ["model-b"]
    capsys.readouterr()

    # status --json is untouched: same key set, and it sees the new state.
    assert cmd_model_routing(
        _ns(routing_command="status", json=True, provider="prov")
    ) == 0
    report = _parse(capsys.readouterr().out)
    assert set(report) == REPORT_KEYS
    assert report["enabled"] is True
    assert report["settings"]["strategy"] == "quality"
    assert report["settings"]["preferred_models_only"] is True
    assert report["skip_reason"] != "disabled"


def test_configure_help_row_is_listed_under_routing():
    parser, _picker, _routing = _build()
    model_parser = next(
        action for action in parser._subparsers._group_actions  # noqa: SLF001
    ).choices["model"]
    routing_parser = next(
        action for action in model_parser._subparsers._group_actions  # noqa: SLF001
    ).choices["routing"]
    configure = next(
        action for action in routing_parser._subparsers._group_actions  # noqa: SLF001
    ).choices["configure"]
    assert "configure" in routing_parser.format_help()
    assert configure.description


def test_configure_flags_do_not_widen_the_older_verbs():
    parser, _picker, _routing = _build()
    for argv in (
        ["model", "routing", "status", "--models", "a"],
        ["model", "routing", "enable", "--models", "a"],
        ["model", "routing", "disable", "--yes"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


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


# ---------------------------------------------------------------------------
# `status --json` — the shared machine-readable contract
# ---------------------------------------------------------------------------
#
# Both trees must emit the same document for the same config: exactly these
# top-level keys, json.dumps(indent=2, sort_keys=True), exit 0 unless the config
# itself cannot be read (then exit 2).

REPORT_KEYS = {
    "enabled", "settings", "candidate_source", "request", "pricing", "selected",
    "selected_provider", "reason", "skip_reason", "current", "kept_current",
    "eligible_count", "candidates",
}


def _parse(out: str):
    import json

    return json.loads(out)


def test_status_json_flag_is_registered_only_on_status():
    parser, _picker, _routing = _build()
    ns = parser.parse_args(["model", "routing", "status", "--json"])
    assert ns.json is True
    # explain keeps its own surface; the flag is not silently accepted there
    with pytest.raises(SystemExit):
        parser.parse_args(["model", "routing", "explain", "hi", "--json"])


def test_status_json_emits_the_contract_document(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status", json=True, provider="prov")) == 0
    out = capsys.readouterr().out
    report = _parse(out)

    assert set(report) == REPORT_KEYS
    assert set(report["settings"]) == {
        "strategy", "stability_threshold", "allow_unknown_metadata",
        "cost_optimization", "preferred_models_only",
    }
    assert set(report["request"]) == {
        "tier", "est_input_tokens", "est_output_tokens", "needs_tools",
        "needs_vision", "needs_audio", "reasons",
    }
    assert set(report["pricing"]) == {"cheapest_eligible", "median_fallback", "policy"}
    assert report["settings"]["stability_threshold"] == pytest.approx(0.15)
    # sort_keys + indent=2: re-serialising reproduces the output byte for byte
    import json

    assert out.strip() == json.dumps(report, indent=2, sort_keys=True)


def test_status_json_reports_disabled_but_still_shows_the_hypothetical(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status", json=True, provider="prov")) == 0
    report = _parse(capsys.readouterr().out)
    assert report["enabled"] is False
    assert report["skip_reason"] == "disabled"
    # the hypothetical selection is still computed and explained
    assert report["candidates"]
    assert report["selected"] in {"model-a", "model-b"}
    assert report["reason"]


def test_status_json_is_read_only(home, capsys):
    from hermes_cli.config import load_config
    from hermes_cli.main import cmd_model_routing

    before = repr(load_config())
    cmd_model_routing(_ns(routing_command="status", json=True, provider="prov"))
    capsys.readouterr()
    assert repr(load_config()) == before


def test_status_json_models_entries_carry_metadata(home, capsys, monkeypatch):
    """`models[]` is the canonical list; its metadata reaches the report."""
    import hermes_cli.config as cfg

    (home / "config.yaml").write_text(
        "model:\n"
        "  default: model-a\n"
        "  provider: prov\n"
        "smart_model_routing:\n"
        "  enabled: true\n"
        "  models:\n"
        "    - name: model-a\n"
        "      provider: prov\n"
        "      cost_in: 1.0\n"
        "      cost_out: 2.0\n"
        "      tier_rank: 2\n"
        "      supports_tools: true\n"
        "    - name: model-b\n"
        "      provider: prov\n"
        "      cost_in: 10.0\n"
        "      cost_out: 10.0\n"
        "      tier_rank: 4\n"
        "      supports_tools: true\n"
        "    - name: mystery\n"
        "      provider: prov\n",
        encoding="utf-8",
    )
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()

    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status", json=True)) == 0
    report = _parse(capsys.readouterr().out)

    assert report["candidate_source"] == "models"
    assert [c["model"] for c in report["candidates"]] == ["model-a", "model-b", "mystery"]
    assert report["eligible_count"] == 3
    # the unpriced model is charged the median of the two known prices
    mystery = next(c for c in report["candidates"] if c["model"] == "mystery")
    assert mystery["priced"] is False
    assert mystery["charged_as"] == "median"
    assert mystery["cost_in"] == pytest.approx(5.5)


def test_status_json_prompt_argument_classifies_the_request(home, capsys):
    from hermes_cli.main import cmd_model_routing

    argv = [
        "Prove", "the", "invariant", "and", "derive", "the", "trade-offs",
        "```python", "x = 1", "```",
    ]
    assert cmd_model_routing(
        _ns(routing_command="status", json=True, provider="prov", prompt=argv)
    ) == 0
    report = _parse(capsys.readouterr().out)
    assert report["request"]["tier"] == "advanced"
    assert report["request"]["est_output_tokens"] > 300


def test_status_json_exits_two_when_the_config_cannot_be_read(monkeypatch, capsys):
    import hermes_cli.config as cfg

    def _boom(*_a, **_kw):
        raise RuntimeError("config.yaml is not readable")

    monkeypatch.setattr(cfg, "load_config", _boom)
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status", json=True)) == 2
    captured = capsys.readouterr()
    assert "not readable" in captured.err
    assert captured.out == ""


def test_status_json_exits_zero_for_a_misconfigured_but_readable_config(
    tmp_path, monkeypatch, capsys
):
    """Garbage in the section must be reported, never crash the command."""
    home = tmp_path / "broken"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    (home / "config.yaml").write_text(
        "model:\n  provider: prov\n"
        "smart_model_routing:\n"
        "  enabled: true\n"
        "  models: not-a-list\n"
        "  stability_threshold: abc\n"
        "  strategy: nonsense\n",
        encoding="utf-8",
    )
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status", json=True)) == 0
    report = _parse(capsys.readouterr().out)
    assert set(report) == REPORT_KEYS
    assert report["settings"]["strategy"] == "automatic"
    assert report["settings"]["stability_threshold"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# catalog refresh (directive 5.1: reuse the existing cache, never a second one)
# ---------------------------------------------------------------------------

def test_refresh_verb_reaches_the_routing_handler():
    parser, _picker, routing = _build()
    ns = parser.parse_args(["model", "routing", "refresh"])
    assert ns.func is routing
    assert ns.routing_command == "refresh"


def test_refresh_force_refreshes_the_engine_catalog_and_writes_no_config(
    home, monkeypatch, capsys
):
    """The refresh must hit the SAME registry the engine reads, and only that."""
    import agent.models_dev as models_dev
    from hermes_cli.main import cmd_model_routing

    before = (home / "config.yaml").read_bytes()
    seen = []

    def _fake_fetch(force_refresh=False, **kwargs):
        seen.append(force_refresh)
        return {"prov": {"models": {}}, "other": {"models": {}}}

    monkeypatch.setattr(models_dev, "fetch_models_dev", _fake_fetch)
    monkeypatch.setattr(
        models_dev, "list_provider_models",
        lambda provider, **kw: ["model-a", "model-c"] if provider == "prov" else [],
    )

    assert cmd_model_routing(_ns(routing_command="refresh")) == 0
    out = capsys.readouterr().out
    assert seen == [True], "refresh must force a catalog fetch, not serve the cache"
    assert "providers:  2" in out
    assert "2 models listed" in out
    assert (home / "config.yaml").read_bytes() == before, "refresh is not a config write"


def test_refresh_says_so_when_the_catalog_cannot_be_fetched(home, monkeypatch, capsys):
    """An unreachable catalog is reported, never dressed up as a success."""
    import agent.models_dev as models_dev
    from hermes_cli.main import cmd_model_routing

    monkeypatch.setattr(
        models_dev, "fetch_models_dev", lambda force_refresh=False, **kw: {}
    )
    rc = cmd_model_routing(_ns(routing_command="refresh"))
    captured = capsys.readouterr()
    assert rc == 1
    assert "could not refresh" in captured.err
    assert "✓" not in captured.out


def test_refresh_json_reports_the_same_facts(home, monkeypatch, capsys):
    import json

    import agent.models_dev as models_dev
    from hermes_cli.main import cmd_model_routing

    monkeypatch.setattr(
        models_dev, "fetch_models_dev",
        lambda force_refresh=False, **kw: {"prov": {"models": {}}},
    )
    monkeypatch.setattr(
        models_dev, "list_provider_models", lambda provider, **kw: ["model-a"]
    )
    assert cmd_model_routing(_ns(routing_command="refresh", json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refreshed"] is True
    assert payload["source"] == "models.dev"
    assert payload["provider"] == "prov"
    assert payload["models"] == 1
    assert payload["config_written"] is False


# ---------------------------------------------------------------------------
# bare `hermes model routing` — the `hermes model` convention
# ---------------------------------------------------------------------------

def test_bare_routing_is_interactive_when_a_human_is_present(home, monkeypatch):
    import hermes_cli.main as main

    called = []
    monkeypatch.setattr(main, "_routing_is_tty", lambda: True)
    monkeypatch.setattr(
        main, "_routing_configure_interactive",
        lambda args, config: called.append(config) or 0,
    )
    monkeypatch.setattr(
        main, "_routing_configure_common",
        lambda args, config: pytest.fail("bare routing must not take the flag path"),
    )
    assert main.cmd_model_routing(_ns()) == 0
    assert len(called) == 1


def test_bare_routing_prints_status_when_nobody_can_answer(home, monkeypatch, capsys):
    import hermes_cli.main as main

    monkeypatch.setattr(main, "_routing_is_tty", lambda: False)
    monkeypatch.setattr(
        main, "_routing_configure_interactive",
        lambda args, config: pytest.fail("no TTY, so no prompt"),
    )
    assert main.cmd_model_routing(_ns()) == 0
    assert "Dynamic model selection" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the renderer must not claim a write, and the offer must name the real state
# ---------------------------------------------------------------------------

def test_status_never_claims_a_saved_block_that_is_not_in_the_file(home, capsys):
    """No block in config.yaml ⇒ no 'Saved in' footer (it would be a false claim)."""
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="status")) == 0
    assert "Saved in:" not in capsys.readouterr().out


def test_status_reports_the_block_once_the_file_has_one(home, capsys):
    from hermes_cli.main import cmd_model_routing

    assert cmd_model_routing(_ns(routing_command="enable")) == 0
    capsys.readouterr()
    assert cmd_model_routing(_ns(routing_command="status")) == 0
    assert "Saved in: config.yaml (smart_model_routing)" in capsys.readouterr().out


def test_interactive_offer_names_the_current_state_not_its_opposite(home, monkeypatch):
    """The 'keep it as it is' option must read as the state the user is in."""
    import hermes_cli.main as main
    from hermes_cli.config import load_config

    seen = []

    def _choice(question, choices, default=0):
        seen.append((question, list(choices)))
        return None  # cancel at the first prompt: nothing may be written

    monkeypatch.setattr(main, "_routing_ask_choice", _choice)

    main._routing_configure_interactive(_ns(), load_config())
    assert seen[0][1][0] == "Keep it off", seen[0]

    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    from agent import model_router as router

    router.update_routing({"enabled": True})
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    seen.clear()
    main._routing_configure_interactive(_ns(), load_config())
    assert seen[0][1][0] == "Keep it on", seen[0]
