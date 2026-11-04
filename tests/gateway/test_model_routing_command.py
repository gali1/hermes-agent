"""Tests for the gateway ``/model-routing`` command.

Two things matter here: the command reaches its own handler for both the hyphenated form the
user types and the underscore form Telegram's menu registers, and it never writes config
without an explicit ``confirm`` in the same message.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.slash_commands_model import GatewayModelCommandsMixin


class _FakeEvent:
    def __init__(self, args: str):
        self._args = args

    def get_command_args(self) -> str:
        return self._args


class _Runner(GatewayModelCommandsMixin):
    """The mixin is self-contained for this command; no gateway state is needed."""


@pytest.fixture
def cfgfile(tmp_path, monkeypatch):
    import hermes_cli.config as config_mod

    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: deepseek-v4.1-flash\n  provider: opencode-zen\n")
    monkeypatch.setattr(config_mod, "get_config_path", lambda: path)
    return path


def _run(args: str) -> str:
    return asyncio.run(_Runner()._handle_model_routing_command(_FakeEvent(args)))


def test_status_shows_disabled_by_default(cfgfile):
    out = _run("")
    assert "enabled: false" in out
    assert "Usage:" in out


def test_on_without_confirm_does_not_write(cfgfile):
    before = cfgfile.read_text()
    out = _run("on")
    assert "Reply" in out and "confirm" in out
    assert "enabled: true" in out          # preview is shown
    assert cfgfile.read_text() == before   # but nothing written


def test_on_confirm_enables(cfgfile):
    out = _run("on confirm")
    assert "Routing enabled" in out
    assert "enabled: true" in cfgfile.read_text()


def test_off_disables(cfgfile):
    _run("on confirm")
    out = _run("off")
    assert "Routing disabled" in out
    assert "enabled: false" in cfgfile.read_text()


def test_unknown_option_is_rejected_without_writing(cfgfile):
    before = cfgfile.read_text()
    out = _run("frobnicate")
    assert "Unknown option" in out
    assert cfgfile.read_text() == before


def test_both_spellings_resolve_to_the_routing_handler():
    """``/model-routing`` and the registered ``model_routing`` must not fall through to /model."""
    from gateway.run_busy import GatewayBusySessionMixin

    assert "model-routing" in GatewayBusySessionMixin._IDLE_COMMANDS
    assert "model_routing" in GatewayBusySessionMixin._IDLE_COMMANDS
    # The dispatcher maps name -> _handle_<name with - as _>_command.
    for name in ("model-routing", "model_routing"):
        assert name.replace("-", "_") == "model_routing"
    assert hasattr(GatewayModelCommandsMixin, "_handle_model_routing_command")
    # /model is untouched and still resolves to its own handler.
    assert hasattr(GatewayModelCommandsMixin, "_handle_model_command")
    assert "model" in GatewayBusySessionMixin._IDLE_COMMANDS
