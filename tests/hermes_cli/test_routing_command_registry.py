"""The routing command must be discoverable and reachable under both spellings.

Telegram's command menu accepts only ``[a-z0-9_]``, so the menu registers ``model_routing``
while users type ``/model-routing``. Both must resolve to the same command, and ``/model``
must keep resolving to itself.
"""

from __future__ import annotations

from hermes_cli.commands import resolve_command
from hermes_cli.commands_platforms import telegram_bot_commands


def test_both_spellings_resolve_to_the_routing_command():
    for typed in ("model-routing", "model_routing", "MODEL-ROUTING"):
        cmd = resolve_command(typed)
        assert cmd is not None, typed
        assert cmd.name == "model-routing"


def test_model_is_unaffected():
    cmd = resolve_command("model")
    assert cmd is not None and cmd.name == "model"


def test_telegram_menu_offers_the_underscore_form():
    """The menu is the only way a user discovers it without knowing the name."""
    names = [name for name, _desc in telegram_bot_commands()]
    assert "model_routing" in names
    assert "model" in names


def test_menu_description_is_not_empty():
    descriptions = dict(telegram_bot_commands())
    assert descriptions["model_routing"].strip()
