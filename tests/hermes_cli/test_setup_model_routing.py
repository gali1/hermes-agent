"""The OpenCode routing step in the model-setup flow (M3).

Behaviour contracts, not snapshots:

* the step is offered only for the OpenCode family, directly after the model
  choice, and only on an interactive stdin;
* the default answer (skip) writes nothing — ``config.yaml`` stays
  byte-identical;
* picking models persists through ``agent.model_router.update_routing`` and is
  visible in ``routing_settings()`` read back from disk;
* a non-interactive / CI stdin returns before any prompt;
* re-running setup preserves an existing ``smart_model_routing`` block instead
  of resetting it.
"""

from __future__ import annotations

import pytest

MODELS = ["alpha", "beta", "gamma"]
SEED_CONFIG = "# user comment kept by the round-trip writer\nmodel:\n  default: alpha\n  provider: opencode-zen\n"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch HERMES_HOME (never the real ~/.hermes) with a minimal config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    path = config_mod.get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SEED_CONFIG, encoding="utf-8")
    yield path
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()


def _interactive(monkeypatch, answers, picks=()):
    """Wire the wizard to an interactive stdin with scripted radio answers and picks."""
    from hermes_cli import model_setup_flows_common as common

    monkeypatch.setattr("hermes_cli.setup.is_interactive_stdin", lambda: True)
    queue = list(answers)
    monkeypatch.setattr(
        common, "_curses_choice",
        lambda title, rows, default: queue.pop(0) if queue else default)
    picks = list(picks)

    def fake_picker(model_ids, current_model="", **kwargs):
        return picks.pop(0) if picks else None

    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", fake_picker)
    return common


def _never_prompt(monkeypatch):
    """Fail loudly if the wizard prompts at all."""
    from hermes_cli import model_setup_flows_common as common

    def boom(*_a, **_k):
        raise AssertionError("the routing step must not prompt here")

    monkeypatch.setattr(common, "_curses_choice", boom)
    monkeypatch.setattr(common, "_ask", boom)
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", boom)


# ---------------------------------------------------------------------------
# (a) skip is the default and writes nothing
# ---------------------------------------------------------------------------


def test_default_skip_leaves_config_byte_identical(home, monkeypatch):
    common = _interactive(monkeypatch, answers=[0])  # 0 = the default row
    before = home.read_bytes()

    result = common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS)

    assert result is None
    assert home.read_bytes() == before
    from agent.model_router import routing_settings

    assert routing_settings()["enabled"] is False


def test_cancel_writes_nothing(home, monkeypatch):
    common = _interactive(monkeypatch, answers=[-1])  # curses cancel
    before = home.read_bytes()

    assert common.offer_model_routing(provider_id="opencode-go", model_list=MODELS) is None
    assert home.read_bytes() == before


def test_numbered_fallback_default_is_skip(home, monkeypatch):
    """No curses (piped terminal): the numbered prompt's empty answer means skip."""
    from hermes_cli import model_setup_flows_common as common

    monkeypatch.setattr("hermes_cli.setup.is_interactive_stdin", lambda: True)
    monkeypatch.setattr(common, "_curses_choice", lambda *a, **k: None)
    monkeypatch.setattr(common, "_ask", lambda *a, **k: "")
    before = home.read_bytes()

    assert common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS) is None
    assert home.read_bytes() == before


def test_unknown_pick_is_rejected_with_a_message(home, monkeypatch, capsys):
    common = _interactive(monkeypatch, answers=[2], picks=["delta", "alpha", "beta", None])

    common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS)

    assert "'delta' is not one of" in capsys.readouterr().out
    from agent.model_router import routing_settings

    assert routing_settings()["models"] == ["alpha", "beta"]


def test_single_discovered_model_offers_nothing(home, monkeypatch):
    common = _interactive(monkeypatch, answers=[])
    _never_prompt(monkeypatch)
    before = home.read_bytes()

    assert common.offer_model_routing(provider_id="opencode-zen", model_list=["alpha"]) is None
    assert home.read_bytes() == before


# ---------------------------------------------------------------------------
# (b) a selection persists through update_routing and is visible on disk
# ---------------------------------------------------------------------------


def test_selected_models_are_persisted_and_read_back(home, monkeypatch, capsys):
    common = _interactive(monkeypatch, answers=[2], picks=["alpha", "beta", None])
    before = home.read_text(encoding="utf-8")

    result = common.offer_model_routing(
        provider_id="opencode-zen", model_list=MODELS, current_model="alpha")

    assert result is not None and result["written"] is True
    from agent.model_router import routing_settings

    settings = routing_settings()
    assert settings["enabled"] is True
    assert settings["preferred_models_only"] is True
    assert settings["models"] == ["alpha", "beta"]
    # the pre-existing YAML key and its comment survive the write
    after = home.read_text(encoding="utf-8")
    assert "provider: opencode-zen" in after
    assert "# user comment kept by the round-trip writer" in after
    assert before.count("model:") == after.count("model:")
    out = capsys.readouterr().out
    assert "Smart model routing enabled" in out
    assert "hermes model routing status" in out


def test_all_eligible_writes_no_model_restriction(home, monkeypatch):
    common = _interactive(monkeypatch, answers=[1])

    result = common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS)

    assert result is not None
    from agent.model_router import routing_settings

    settings = routing_settings()
    assert settings["enabled"] is True
    assert settings["preferred_models_only"] is False
    assert settings["models"] == []


def test_fewer_than_two_picks_saves_nothing(home, monkeypatch):
    common = _interactive(monkeypatch, answers=[2], picks=["alpha", None])
    before = home.read_bytes()

    assert common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS) is None
    assert home.read_bytes() == before


# ---------------------------------------------------------------------------
# (c) non-interactive stdin returns immediately
# ---------------------------------------------------------------------------


def test_non_interactive_returns_without_prompting(home, monkeypatch):
    from hermes_cli import model_setup_flows_common as common

    monkeypatch.setattr("hermes_cli.setup.is_interactive_stdin", lambda: False)
    _never_prompt(monkeypatch)
    before = home.read_bytes()

    assert common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS) is None
    assert home.read_bytes() == before


# ---------------------------------------------------------------------------
# (d) other providers never see the step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_id", ["zai", "nous", "openrouter", "custom:mine", ""])
def test_non_opencode_provider_never_prompts(home, monkeypatch, provider_id):
    from hermes_cli import model_setup_flows_common as common

    monkeypatch.setattr("hermes_cli.setup.is_interactive_stdin", lambda: True)
    _never_prompt(monkeypatch)
    before = home.read_bytes()

    assert common.offer_model_routing(provider_id=provider_id, model_list=MODELS) is None
    assert home.read_bytes() == before


# ---------------------------------------------------------------------------
# (e) a re-run preserves the existing block
# ---------------------------------------------------------------------------


def test_rerun_keeps_existing_block_by_default(home, monkeypatch):
    from agent.model_router import routing_settings, update_routing

    update_routing({"enabled": True, "preferred_models_only": True, "models": ["alpha", "beta"]})
    before = home.read_bytes()

    common = _interactive(monkeypatch, answers=[0])  # "Keep the current routing configuration"
    assert common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS) is None

    assert home.read_bytes() == before
    settings = routing_settings()
    assert settings["enabled"] is True
    assert settings["models"] == ["alpha", "beta"]


def test_rerun_modify_starts_from_existing_models(home, monkeypatch):
    from agent.model_router import routing_settings, update_routing

    update_routing({"enabled": True, "preferred_models_only": True, "models": ["alpha", "beta"]})

    # 1 = change, 1 = "choose the models" (the current mode), then add gamma only
    common = _interactive(monkeypatch, answers=[1, 1], picks=["gamma", None])
    result = common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS)

    assert result is not None
    settings = routing_settings()
    assert settings["enabled"] is True
    assert settings["models"] == ["alpha", "beta", "gamma"]


def test_rerun_can_turn_routing_off(home, monkeypatch):
    from agent.model_router import routing_settings, update_routing

    update_routing({"enabled": True, "preferred_models_only": False})
    common = _interactive(monkeypatch, answers=[2])  # keep / change / turn off

    result = common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS)

    assert result is not None
    assert routing_settings()["enabled"] is False


def test_rerun_keeps_models_missing_from_todays_discovery(home, monkeypatch, capsys):
    from agent.model_router import routing_settings, update_routing

    update_routing({"enabled": True, "preferred_models_only": True,
                    "models": ["alpha", "retired-model"]})
    common = _interactive(monkeypatch, answers=[1, 1], picks=["beta", None])

    common.offer_model_routing(provider_id="opencode-zen", model_list=MODELS)

    assert routing_settings()["models"] == ["alpha", "retired-model", "beta"]
    assert "retired-model" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the flow itself: called after the model selection, OpenCode only
# ---------------------------------------------------------------------------


def _drive_flow(monkeypatch, provider_id, log):
    from hermes_cli import model_setup_flows as flows
    from hermes_cli import model_setup_flows_common as common

    monkeypatch.setattr(flows, "_ensure_flow_api_key", lambda pid, pconfig, **kw: ("key", "key", False))
    monkeypatch.setattr(flows, "_prompt_base_url_override", lambda base, env, **kw: base)
    monkeypatch.setattr(flows, "_select_zai_endpoint", lambda base: base)
    monkeypatch.setattr(flows, "_api_key_provider_model_list", lambda *a, **k: list(MODELS))
    monkeypatch.setattr(flows, "_pick_model_or_prompt", lambda models, prompt, **kw: models[0])
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *a, **k: {})
    monkeypatch.setattr(flows, "_finish_model", lambda *a, **k: log.append("model"))
    monkeypatch.setattr(common, "offer_model_routing",
                        lambda **kw: log.append(("routing", kw.get("provider_id"), kw.get("model_list"))))
    flows._model_flow_api_key_provider({}, provider_id)


def test_opencode_flow_offers_routing_right_after_the_model_choice(monkeypatch):
    log = []
    _drive_flow(monkeypatch, "opencode-zen", log)

    assert log == ["model", ("routing", "opencode-zen", MODELS)]


def test_other_provider_flow_never_offers_routing(monkeypatch):
    log = []
    _drive_flow(monkeypatch, "zai", log)

    assert log == ["model"]
