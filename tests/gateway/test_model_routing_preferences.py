"""``/model-routing`` preferences — every setting the engine reads, from chat.

Directive §4: each routing setting the runtime reads must be reachable from the
interfaces, and each control must map to something real.  This module covers the
chat half of the second-surface work:

* ``allow_unknown_metadata``, ``exclude``, ``weights`` and ``reliability`` are
  settable from the preferences menu, through the same
  ``agent.model_router.update_routing`` writer the CLI uses (C12).
* A typed value is parsed by ``agent.model_router`` and validated by
  ``validate_routing`` — not by a gateway-local parser — so a payload the chat
  accepts is one ``hermes model routing configure`` accepts.
* Nothing is written before the final Save: the file is byte-identical while
  values are pending, and after Cancel.
* ``llm_assisted_classification`` no longer has a control: the engine parses the
  key but nothing reads it, so offering it would be a button that does nothing.
"""

from __future__ import annotations

import time

import pytest

import gateway.run as gateway_run
from agent import model_router as mr
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
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(ROUTED_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "1")
    return home


@pytest.fixture
def runner():
    instance = object.__new__(gateway_run.GatewayRunner)
    instance.adapters = {}
    return instance


def _event(text: str, *, user_id: str = "1", chat_id: str = "2") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id=user_id, chat_id=chat_id, user_name="tester",
    )
    return MessageEvent(text=text, source=source)


async def _dispatch(runner, text: str, **kwargs) -> str:
    event = _event(text, **kwargs)
    return (await runner._handle_model_routing_command(event)) or ""


def _config_text(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


async def _open_prefs(runner) -> str:
    await _dispatch(runner, "/model-routing")
    return await _dispatch(runner, "/model-routing 4")


# ---------------------------------------------------------------- the menu


class TestPreferencesMenu:
    @pytest.mark.asyncio
    async def test_lists_every_setting_the_engine_reads(self, routing_home, runner):
        prefs = await _open_prefs(runner)
        for row in (
            "Strategy", "Stability threshold", "Cost optimisation",
            "Mid-session cost downgrade", "Route only among my selected models",
            "Allow unknown capability metadata", "Never auto-select",
            "Scoring weights", "Declared reliability",
        ):
            assert row in prefs, row
        assert "10 Back to the main menu" in prefs

    @pytest.mark.asyncio
    async def test_no_control_for_a_setting_nothing_reads(self, routing_home, runner):
        prefs = await _open_prefs(runner)
        assert "LLM-assisted" not in prefs

    @pytest.mark.asyncio
    async def test_every_menu_row_maps_to_a_key_the_engine_reads(self, routing_home, runner):
        """A row exists ⇔ RoutingConfig reads that key."""
        prefs = await _open_prefs(runner)
        advertised = {
            "strategy": "Strategy", "stability_threshold": "Stability threshold",
            "cost_optimization": "Cost optimisation",
            "switch_on_cost": "Mid-session cost downgrade",
            "preferred_models_only": "Route only among my selected models",
            "allow_unknown_metadata": "Allow unknown capability metadata",
            "exclude": "Never auto-select", "weights": "Scoring weights",
            "reliability": "Declared reliability",
        }
        rad = mr.RoutingConfig()
        for key, label in advertised.items():
            assert label in prefs, label
            assert hasattr(rad, key), key


# ---------------------------------------------------------------- toggles


class TestToggles:
    @pytest.mark.asyncio
    async def test_allow_unknown_metadata_toggles_and_is_pending(self, routing_home, runner):
        before = _config_text(routing_home)
        await _open_prefs(runner)
        reply = await _dispatch(runner, "/model-routing 6")

        assert "allow_unknown_metadata: on → off" in reply
        assert _config_text(routing_home) == before       # nothing written yet

        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")       # enable -> confirm
        await _dispatch(runner, "/model-routing yes")
        assert mr.routing_settings()["allow_unknown_metadata"] is False

    @pytest.mark.asyncio
    async def test_switch_on_cost_toggle_still_works(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 4")       # mid-session downgrade
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")
        assert mr.routing_settings()["switch_on_cost"] is True


# ---------------------------------------------------------------- typed values


class TestTypedValues:
    @pytest.mark.asyncio
    async def test_exclude_step_records_a_priority_list(self, routing_home, runner):
        before = _config_text(routing_home)
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 7")
        reply = await _dispatch(runner, "/model-routing exclude openai/gpt-5-nano,anthropic/claude-haiku")
        assert "exclude: (none) → openai/gpt-5-nano, anthropic/claude-haiku" in reply
        assert _config_text(routing_home) == before

        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")
        assert mr.routing_settings()["exclude"] == [
            "openai/gpt-5-nano", "anthropic/claude-haiku",
        ]

    @pytest.mark.asyncio
    async def test_exclude_none_clears_the_list(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 7")
        await _dispatch(runner, "/model-routing exclude openai/gpt-5-nano")
        await _dispatch(runner, "/model-routing 7")
        await _dispatch(runner, "/model-routing exclude none")
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")
        assert mr.routing_settings()["exclude"] == []

    @pytest.mark.asyncio
    async def test_weights_step_merges_only_the_named_dimensions(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 8")
        reply = await _dispatch(runner, "/model-routing weights capability=0.5,cost=0.3")
        assert "weights:" in reply
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")

        settings = mr.routing_settings()
        # 0.5 and 0.3 were authored; the other three kept their defaults, and the
        # whole mapping is normalised on read.
        assert settings["weights"]["capability"] == pytest.approx(0.5 / 1.05, abs=1e-4)
        assert settings["weights"]["cost"] == pytest.approx(0.3 / 1.05, abs=1e-4)
        assert settings["weights"]["continuity"] == pytest.approx(0.05 / 1.05, abs=1e-4)

    @pytest.mark.asyncio
    async def test_weights_default_resets(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 8")
        await _dispatch(runner, "/model-routing weights capability=0.9")
        await _dispatch(runner, "/model-routing 8")
        await _dispatch(runner, "/model-routing weights default")
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")
        assert mr.routing_settings()["weights"] == pytest.approx(mr.default_weights())

    @pytest.mark.asyncio
    async def test_reliability_step_sets_and_clears(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 9")
        await _dispatch(runner, "/model-routing reliability openai/gpt-5-mini=0.97")
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")
        assert mr.routing_settings()["reliability"] == {
            "openai/gpt-5-mini": pytest.approx(0.97),
        }

    @pytest.mark.asyncio
    async def test_an_unparsable_value_is_refused_and_stays_pending_free(self, routing_home, runner):
        before = _config_text(routing_home)
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 8")
        reply = await _dispatch(runner, "/model-routing weights capability")
        assert "must look like name=value" in reply
        assert "Pending (not saved yet)" not in reply
        assert _config_text(routing_home) == before

    @pytest.mark.asyncio
    async def test_an_unknown_dimension_is_refused_by_the_validator(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 8")
        reply = await _dispatch(runner, "/model-routing weights bird=0.4")
        assert "not a scoring dimension" in reply
        assert "smart_model_routing.weights.bird" in reply
        assert mr.routing_settings()["weights"] == pytest.approx(mr.default_weights())

    @pytest.mark.asyncio
    async def test_an_out_of_range_rate_is_refused(self, routing_home, runner):
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 9")
        reply = await _dispatch(runner, "/model-routing reliability openai/gpt-5-mini=1.4")
        assert "between 0 and 1" in reply


# ---------------------------------------------------------------- save / cancel


class TestSaveAndCancel:
    @pytest.mark.asyncio
    async def test_cancel_after_advanced_edits_leaves_the_file_byte_identical(
            self, routing_home, runner):
        before = _config_text(routing_home)
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 6")
        await _dispatch(runner, "/model-routing 7")
        await _dispatch(runner, "/model-routing exclude openai/gpt-5-nano")
        await _dispatch(runner, "/model-routing 9")
        await _dispatch(runner, "/model-routing reliability openai/gpt-5-mini=0.9")
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")       # enable -> confirm
        reply = await _dispatch(runner, "/model-routing cancel")
        assert "Cancelled" in reply
        assert _config_text(routing_home) == before

    @pytest.mark.asyncio
    async def test_an_abandoned_flow_times_out_and_applies_nothing(self, routing_home, runner):
        before = _config_text(routing_home)
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 6")
        key = next(iter(runner._model_routing_flows()))
        runner._model_routing_flows()[key]["ts"] = time.time() - 10_000
        reply = await _dispatch(runner, "/model-routing yes")
        assert "timed out" in reply.lower()
        assert _config_text(routing_home) == before

    @pytest.mark.asyncio
    async def test_saved_values_are_readable_from_the_cli_surface(self, routing_home, runner):
        """Cross-interface: what chat wrote is what ``hermes model routing status`` reads."""
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 6")
        await _dispatch(runner, "/model-routing 7")
        await _dispatch(runner, "/model-routing exclude openai/gpt-5-nano")
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        reply = await _dispatch(runner, "/model-routing yes")

        cli_view = mr.routing_settings()          # no argument: reads config.yaml itself
        assert cli_view["exclude"] == ["openai/gpt-5-nano"]
        assert cli_view["allow_unknown_metadata"] is False
        assert "Never auto-select: openai/gpt-5-nano" in reply


# ---------------------------------------------------------------- one writer (C12)


class TestOneWriter:
    @pytest.mark.asyncio
    async def test_every_preferences_write_uses_update_routing(self, routing_home, runner,
                                                               monkeypatch):
        seen: list[dict] = []
        real = mr.update_routing

        def _spy(changes, **kwargs):
            seen.append(dict(changes))
            return real(changes, **kwargs)

        monkeypatch.setattr(mr, "update_routing", _spy)
        await _open_prefs(runner)
        await _dispatch(runner, "/model-routing 6")
        await _dispatch(runner, "/model-routing 7")
        await _dispatch(runner, "/model-routing exclude openai/gpt-5-nano")
        await _dispatch(runner, "/model-routing 8")
        await _dispatch(runner, "/model-routing weights capability=0.5")
        await _dispatch(runner, "/model-routing 9")
        await _dispatch(runner, "/model-routing reliability openai/gpt-5-mini=0.95")
        await _dispatch(runner, "/model-routing 10")
        await _dispatch(runner, "/model-routing 1")
        await _dispatch(runner, "/model-routing yes")

        assert len(seen) == 1, seen                 # exactly one write, at Save
        assert set(seen[0]) >= {
            "enabled", "allow_unknown_metadata", "exclude", "weights", "reliability",
        }
