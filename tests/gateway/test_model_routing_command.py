"""Gateway ``/model-routing`` (+ ``/model_routing``): registration, dispatch, flow, authz, sync.

The dispatch tests go through the REAL surfaces — ``MessageEvent.get_command`` →
``hermes_cli.commands.resolve_command`` → ``is_gateway_known_command`` → the runner's
``_gateway_plain_command_handlers`` / ``_gateway_idle_command_handlers`` tables — because a
handler that only "exists" is not a command a user can reach.

Every write goes to a scratch ``HERMES_HOME`` (``tmp_path``); the real ``~/.hermes`` is never
touched (tests/home_io_guard.py enforces it).
"""

from __future__ import annotations

import time

import pytest

import gateway.run as gateway_run
from agent import model_router as mr
from hermes_cli.commands import (
    GATEWAY_KNOWN_COMMANDS,
    is_gateway_known_command,
    resolve_command,
)
from hermes_cli.commands_platforms import telegram_menu_commands
from hermes_cli.config import load_config_readonly
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

ROUTED_CONFIG = """\
model:
  default: openai/gpt-5-mini
  provider: openrouter
smart_model_routing:
  models:
    - openai/gpt-5-mini
    - openai/gpt-5-nano
    - anthropic/claude-haiku
"""


@pytest.fixture
def routing_home(tmp_path, monkeypatch):
    """A scratch HERMES_HOME whose config already declares three preferred models."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(ROUTED_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "1")
    return home


@pytest.fixture
def runner():
    """A bare GatewayRunner: the mixin methods under test, no startup side effects."""
    instance = object.__new__(gateway_run.GatewayRunner)
    instance.adapters = {}
    return instance


def _event(text: str, *, user_id: str = "1", chat_id: str = "2") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id=user_id, chat_id=chat_id, user_name="tester",
    )
    return MessageEvent(text=text, source=source)


async def _dispatch(runner, text: str, **kwargs) -> str:
    """Resolve and dispatch *text* exactly as the gateway's idle path does."""
    event = _event(text, **kwargs)
    command = event.get_command()
    assert command, f"{text!r} did not parse as a command"
    cmd_def = resolve_command(command)
    assert cmd_def is not None, f"/{command} is not in COMMAND_REGISTRY"
    canonical = cmd_def.name
    assert is_gateway_known_command(canonical), canonical
    handler = (
        runner._gateway_plain_command_handlers().get(canonical)
        or runner._gateway_idle_command_handlers().get(canonical)
    )
    assert handler is not None, f"/{canonical} has no handler in the dispatch tables"
    return await handler(event)


def _config_text(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


# ---------------------------------------------------------------- registration / dispatch


class TestRegistrationAndDispatch:
    def test_both_spellings_resolve_to_the_same_command(self):
        for typed in ("model-routing", "model_routing", "MODEL-ROUTING"):
            resolved = resolve_command(typed)
            assert resolved is not None, typed
            assert resolved.name == "model-routing"

    def test_both_spellings_are_gateway_known(self):
        assert "model-routing" in GATEWAY_KNOWN_COMMANDS
        assert "model_routing" in GATEWAY_KNOWN_COMMANDS

    @pytest.mark.asyncio
    async def test_both_spellings_dispatch_to_the_routing_handler(self, routing_home, runner):
        for typed in ("/model-routing", "/model_routing"):
            event = _event(typed)
            canonical = resolve_command(event.get_command()).name
            assert canonical == "model-routing"
            handler = (
                runner._gateway_plain_command_handlers().get(canonical)
                or runner._gateway_idle_command_handlers().get(canonical)
            )
            assert handler == runner._handle_model_routing_command
            reply = await handler(event)
            assert "Model routing:" in reply
            # Neither spelling may land on /model's handler.
            assert runner._gateway_idle_command_handlers().get(canonical) is not (
                runner._handle_model_command
            )

    @pytest.mark.asyncio
    async def test_model_command_is_unaffected(self, routing_home, runner):
        event = _event("/model")
        assert event.get_command() == "model"
        cmd_def = resolve_command(event.get_command())
        assert cmd_def.name == "model"
        handler = (
            runner._gateway_plain_command_handlers().get("model")
            or runner._gateway_idle_command_handlers().get("model")
        )
        assert handler == runner._handle_model_command
        assert handler != runner._handle_model_routing_command
        # And the argument form still parses as its own command + args.
        arg_event = _event("/model openai/gpt-5-nano")
        assert arg_event.get_command() == "model"
        assert arg_event.get_command_args().strip() == "openai/gpt-5-nano"

    def test_telegram_menu_registers_the_underscore_form(self):
        menu, _hidden = telegram_menu_commands(max_commands=100)
        names = dict(menu)
        assert "model_routing" in names
        assert "model-routing" not in names  # Telegram only accepts [a-z0-9_]
        assert "model" in names

    def test_slack_reaches_it_through_the_hermes_subcommand_map(self):
        # Slack's native slash list is at its 50-command cap, so routing is reachable as
        # /hermes model-routing rather than by evicting an existing command.
        from hermes_cli.commands_platforms import slack_native_slashes, slack_subcommand_map

        assert not {"model-routing", "model_routing"} & {
            name for name, _d, _h in slack_native_slashes()
        }
        mapping = slack_subcommand_map()
        assert "model-routing" in mapping
        assert "model_routing" in mapping

    def test_allowed_while_busy(self, runner):
        # busy dispatch reads _PLAIN_COMMANDS; the idle path reads _IDLE_COMMANDS.
        assert "model-routing" in runner._PLAIN_COMMANDS
        assert "model-routing" in runner._IDLE_COMMANDS
        assert resolve_command("model-routing").busy_policy == "dispatch"


# ---------------------------------------------------------------- read-only view


class TestView:
    @pytest.mark.asyncio
    async def test_view_shows_state_and_actions_without_writing(self, routing_home, runner):
        before = _config_text(routing_home)
        reply = await _dispatch(runner, "/model-routing")
        for expected in ("Model routing: OFF", "Strategy: automatic", "Models:", "Candidates:",
                         "1 Enable routing", "2 Disable routing", "3 Change models",
                         "4 Change preferences", "5 Switch to single-model mode", "6 Close"):
            assert expected in reply, expected
        assert _config_text(routing_home) == before

    @pytest.mark.asyncio
    async def test_view_reflects_a_cli_write_immediately(self, routing_home, runner):
        mr.update_routing({"enabled": True, "strategy": "cost"})
        reply = await _dispatch(runner, "/model-routing")
        assert "Model routing: ON" in reply
        assert "Strategy: cost" in reply


# ---------------------------------------------------------------- the flow


class TestFlow:
    @pytest.mark.asyncio
    async def test_enable_prompts_then_saves_only_on_confirmation(self, routing_home, runner):
        before = _config_text(routing_home)
        await _dispatch(runner, "/model-routing")
        prompt = await _dispatch(runner, "/model-routing 1")
        assert "Save these routing changes?" in prompt
        assert "enabled: off → on" in prompt
        assert _config_text(routing_home) == before, "nothing may be written before confirmation"

        saved = await _dispatch(runner, "/model-routing yes")
        assert "✅ Saved" in saved
        assert "enabled" in saved
        assert "Model routing: ON" in saved
        assert "no gateway restart" in saved.lower()
        assert mr.routing_settings()["enabled"] is True

    @pytest.mark.asyncio
    async def test_disable_and_single_model_mode(self, routing_home, runner):
        mr.update_routing({"enabled": True})
        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 5")
        done = await _dispatch(runner, "/model-routing yes")
        assert "✅ Saved" in done
        assert mr.routing_settings()["enabled"] is False

    @pytest.mark.asyncio
    async def test_cancel_applies_nothing(self, routing_home, runner):
        before = _config_text(routing_home)
        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 1")
        reply = await _dispatch(runner, "/model-routing cancel")
        assert "nothing applied" in reply.lower()
        assert _config_text(routing_home) == before
        assert mr.routing_settings()["enabled"] is False

    @pytest.mark.asyncio
    async def test_abandoned_flow_times_out_and_applies_nothing(self, routing_home, runner):
        before = _config_text(routing_home)
        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 1")
        key = next(iter(runner._model_routing_flows()))
        runner._model_routing_flows()[key]["ts"] = time.time() - 10_000

        reply = await _dispatch(runner, "/model-routing yes")
        assert "timed out" in reply.lower()
        assert _config_text(routing_home) == before
        assert mr.routing_settings()["enabled"] is False

    @pytest.mark.asyncio
    async def test_models_step_writes_the_selection_in_priority_order(self, routing_home, runner):
        await _dispatch(runner, "/model-routing")
        listing = await _dispatch(runner, "/model-routing 3")
        assert "1 openai/gpt-5-mini" in listing
        assert "2 openai/gpt-5-nano" in listing

        prompt = await _dispatch(runner, "/model-routing 2,3")
        assert "Save these routing changes?" in prompt
        assert "preferred_models_only: off → on" in prompt
        await _dispatch(runner, "/model-routing yes")

        settings = mr.routing_settings()
        assert settings["models"] == ["openai/gpt-5-nano", "anthropic/claude-haiku"]
        assert settings["preferred_models_only"] is True

    @pytest.mark.asyncio
    async def test_preferences_step_sets_strategy_and_threshold(self, routing_home, runner):
        await _dispatch(runner, "/model-routing")
        prefs = await _dispatch(runner, "/model-routing 4")
        assert "Change preferences" not in prefs  # sub-menu, not the main menu
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing quality")
        await _dispatch(runner, "/model-routing 2")
        await _dispatch(runner, "/model-routing 0.4")
        prompt = await _dispatch(runner, "/model-routing 10")  # back to menu (7–9 are the advanced settings)
        await _dispatch(runner, "/model-routing 2")  # disable -> confirm
        assert "strategy: automatic → quality" in prompt
        assert "stability_threshold: 0.15 → 0.4" in prompt
        await _dispatch(runner, "/model-routing yes")

        settings = mr.routing_settings()
        assert settings["strategy"] == "quality"
        assert settings["stability_threshold"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_rejected_change_reports_the_reason_and_writes_nothing(self, routing_home, runner, monkeypatch):
        before = _config_text(routing_home)

        def _boom(_changes):
            raise ValueError("smart_model_routing.strategy must be one of automatic, cost, quality")

        monkeypatch.setattr(mr, "update_routing", _boom)
        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 1")
        reply = await _dispatch(runner, "/model-routing yes")
        assert "❌ Not saved" in reply
        assert "strategy" in reply
        assert _config_text(routing_home) == before


# ---------------------------------------------------------------- authorization


class TestAuthorization:
    @pytest.mark.asyncio
    async def test_unauthorized_user_is_refused_and_nothing_is_written(self, tmp_path, monkeypatch, runner):
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(ROUTED_CONFIG, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

        before = _config_text(home)
        reply = await _dispatch(runner, "/model-routing", user_id="1")
        assert "🔒" in reply
        assert "authorized users" in reply
        assert _config_text(home) == before

        # ... and the flow cannot be advanced past the gate either.
        reply = await _dispatch(runner, "/model-routing 1", user_id="1")
        assert "🔒" in reply
        assert _config_text(home) == before
        assert mr.routing_settings()["enabled"] is False

    @pytest.mark.asyncio
    async def test_allowlisted_user_is_admitted(self, tmp_path, monkeypatch, runner):
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(ROUTED_CONFIG, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

        reply = await _dispatch(runner, "/model-routing", user_id="999")
        assert "🔒" not in reply
        assert "Model routing: OFF" in reply


# ---------------------------------------------------------------- cross-interface sync (M5)


class TestCrossInterfaceSync:
    @pytest.mark.asyncio
    async def test_gateway_change_equals_cli_state(self, routing_home, runner):
        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 1")
        reply = await _dispatch(runner, "/model-routing yes")

        # ``routing_settings()`` with no argument is the CLI's view: it loads config.yaml itself.
        cli_settings = mr.routing_settings()
        assert cli_settings["enabled"] is True
        # The gateway's confirmation reports the same block it read back from disk.
        assert "Model routing: ON" in reply
        cli_report = mr.routing_report(load_config_readonly(), "")
        assert cli_report["enabled"] is True
        assert cli_report["settings"]["strategy"] == cli_settings["strategy"]

    def test_update_routing_is_visible_without_a_restart(self, routing_home):
        """route_turn re-reads the block per turn via load_config_readonly (file-signature cached)."""
        assert load_config_readonly().get("smart_model_routing", {}).get("enabled") in (None, False)
        assert mr.routing_preview("hello")["enabled"] is False

        mr.update_routing({"enabled": True, "strategy": "cost"})

        # No restart, no cache bust: the very next read sees the new block.
        live = load_config_readonly()
        assert live["smart_model_routing"]["enabled"] is True
        preview = mr.routing_preview("hello")
        assert preview["enabled"] is True
        assert preview["settings"]["strategy"] == "cost"
        assert preview["skip_reason"] != "disabled"


# ------------------------------------------------- C16/C17 at the gateway surface


BLOCKLESS_CONFIG = """\
model:
  default: openai/gpt-5-mini
  provider: openrouter
"""


@pytest.fixture
def blockless_home(tmp_path, monkeypatch):
    """A scratch HERMES_HOME with **no** ``smart_model_routing`` block (absence == off)."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(BLOCKLESS_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "1")
    return home


class TestNoOpWriteIsNotAClaim:
    """C16 says nothing is written; C17 says the chat cannot claim otherwise.

    Absence of ``smart_model_routing`` is the OFF state, so ``disable`` on a config that
    never carried the block has nothing to do. Until M17 the gateway still answered
    ``✅ Saved smart_model_routing (changed: nothing).`` and promised it would take effect.
    """

    @pytest.mark.asyncio
    async def test_disable_on_a_blockless_config_claims_no_save(self, blockless_home, runner):
        before = _config_text(blockless_home)

        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 2")
        reply = await _dispatch(runner, "/model-routing yes")

        assert "✅ Saved" not in reply, reply
        assert "Nothing saved" in reply
        assert "not modified" in reply
        assert "Takes effect on the next turn" not in reply
        assert "smart_model_routing" not in _config_text(blockless_home)
        assert _config_text(blockless_home) == before

    @pytest.mark.asyncio
    async def test_single_model_mode_is_the_same_action_as_disable(self, blockless_home, runner):
        """§6.3: "switch to single-model mode" is the same action as disable — on the file too."""
        before = _config_text(blockless_home)

        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 5")
        reply = await _dispatch(runner, "/model-routing yes")

        assert "✅ Saved" not in reply, reply
        assert _config_text(blockless_home) == before

    @pytest.mark.asyncio
    async def test_single_model_mode_still_clears_a_real_selection(self, routing_home, runner):
        """The control arm: when the flag IS on, both spellings of disable must write it off."""
        mr.update_routing({"enabled": True, "preferred_models_only": True,
                           "models": ["openai/gpt-5-mini"]})

        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 5")
        reply = await _dispatch(runner, "/model-routing yes")

        assert "✅ Saved" in reply
        settings = mr.routing_settings()
        assert settings["enabled"] is False
        assert settings["preferred_models_only"] is False

    @pytest.mark.asyncio
    async def test_enable_still_reports_a_real_save(self, blockless_home, runner):
        """Control: a change that does write must keep saying so."""
        before = _config_text(blockless_home)

        await _dispatch(runner, "/model-routing")
        await _dispatch(runner, "/model-routing 1")
        reply = await _dispatch(runner, "/model-routing yes")

        assert "✅ Saved smart_model_routing" in reply
        assert "Takes effect on the next turn" in reply
        assert _config_text(blockless_home) != before
        assert mr.routing_settings()["enabled"] is True

    @pytest.mark.asyncio
    async def test_the_button_surface_obeys_the_same_rules(self, blockless_home, runner):
        """C12: the button path is the same applier, so it cannot diverge on the file."""
        before = _config_text(blockless_home)

        await _dispatch(runner, "/model-routing")
        # Same two tokens the inline keyboard sends for "Switch to single-model mode".
        await runner._model_routing_button_action(_event("/model-routing"), "single")
        reply = await runner._model_routing_button_action(_event("/model-routing"), "yes")

        assert "✅ Saved" not in reply, reply
        assert _config_text(blockless_home) == before
