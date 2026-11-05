"""Tests for the gateway ``/model-routing`` command.

Two things matter here: the command reaches its own handler for both the hyphenated form the
user types and the underscore form Telegram's menu registers, and it never writes config
without an explicit ``confirm`` in the same message.

The inline-button path reuses the /model picker's ``send_choice_picker`` mechanism (flat
``cp:<index>`` callbacks), so it is tested at both seams: the handler's choice list and the
Telegram adapter's payload builder (which must stay inside the 64-byte callback cap), and the
adapter's callback handler (a stale or unknown payload must not reach the handler at all).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.slash_commands_model import GatewayModelCommandsMixin


class _FakeEvent:
    def __init__(self, args: str):
        self._args = args
        self.source = "test-source"

    def get_command_args(self) -> str:
        return self._args


class _Runner(GatewayModelCommandsMixin):
    """The mixin is self-contained for this command; authorization is stubbed."""

    def __init__(self, authorized: bool = True):
        self._authorized = authorized

    def _is_user_authorized_for_source(self, source) -> bool:
        return self._authorized


class _Source:
    chat_id = "67890"


class _PickerEvent(_FakeEvent):
    """A button-capable platform event: a real source object, not a stub string."""

    def __init__(self, args: str):
        super().__init__(args)
        self.source = _Source()


class _PickerAdapter:
    """Adapter whose *type* exposes ``send_choice_picker`` — the gate the handler checks."""

    def __init__(self, success: bool = True):
        self.calls: list = []
        self._success = success

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=self._success, message_id="m1")


class _NoPickerAdapter:
    """Adapter with no button capability (text-only platform)."""


class _ButtonRunner(_Runner):
    """Runner wired to a button-capable (or text-only) adapter."""

    def __init__(self, adapter, authorized: bool = True):
        super().__init__(authorized)
        self._adapter = adapter

    def _delivery_adapter_for(self, source):
        return self._adapter

    def _session_key_for_source(self, source) -> str:
        return "test-session"

    def _reply_metadata(self, event):
        return {}


@pytest.fixture
def cfgfile(tmp_path, monkeypatch):
    import hermes_cli.config as config_mod

    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: deepseek-v4.1-flash\n  provider: opencode-zen\n")
    monkeypatch.setattr(config_mod, "get_config_path", lambda: path)
    return path


def _run(args: str, *, authorized: bool = True) -> str:
    return asyncio.run(_Runner(authorized)._handle_model_routing_command(_FakeEvent(args)))


# --------------------------------------------------------------- text-only flow


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


def test_unauthorized_user_is_refused_and_nothing_is_written(cfgfile):
    """This command writes config, so it must fail closed rather than trust the dispatcher."""
    before = cfgfile.read_text()
    out = _run("on confirm", authorized=False)
    assert "not authorized" in out
    assert cfgfile.read_text() == before


# ------------------------------------------------------------ inline buttons


def test_text_only_platform_keeps_the_numbered_usage(cfgfile):
    """An adapter without the button API gets exactly the old text reply."""
    runner = _ButtonRunner(_NoPickerAdapter())
    out = asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    assert "Usage:" in out and "enabled: false" in out


def test_button_platform_gets_the_entry_view(cfgfile):
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)

    out = asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))

    assert out is None                      # picker sent — the adapter owns the response
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert [c["value"] for c in call["choices"]] == ["enable", "disable", "change_models", "cancel"]
    assert "enabled: false" in call["title"]  # the status is the message body
    assert call["chat_id"] == "67890"


def test_on_command_shows_preview_with_confirm_buttons(cfgfile):
    """Typed ``on`` on a button platform still previews — the write needs the second tap."""
    before = cfgfile.read_text()
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)

    out = asyncio.run(runner._handle_model_routing_command(_PickerEvent("on")))

    assert out is None
    call = adapter.calls[0]
    assert [c["value"] for c in call["choices"]] == ["confirm", "cancel"]
    assert "enabled: true" in call["title"]   # preview
    assert cfgfile.read_text() == before      # nothing written yet


def _real_adapter_choice_keyboard(monkeypatch, choices, chat_id="67890"):
    """Build the keyboard the real Telegram adapter would send (PTB is optional in tests)."""
    import plugins.platforms.telegram.adapter as tg

    captured = {}

    class _Button:
        def __init__(self, text, callback_data=None):
            self.text, self.callback_data = text, callback_data

    class _Markup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(tg, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _Markup)

    fake = object.__new__(tg.TelegramAdapter)
    fake._reply_to_mode = None
    fake._choice_picker_state = {}
    fake.format_message = lambda text: text

    async def _send_prompt(_name, _chat_id, _metadata, build, **_kwargs):
        captured["text"], captured["keyboard"], remember = build()
        remember(SimpleNamespace(message_id=1))
        return SendResult(success=True, message_id="1")

    fake._send_prompt = _send_prompt

    async def _on_choice(_chat_id, _value):
        return ""

    asyncio.run(fake.send_choice_picker(
        chat_id=chat_id, title="t", choices=choices, session_key="sk",
        on_choice_selected=_on_choice, metadata=None))
    return captured["keyboard"], fake._choice_picker_state


def test_button_payloads_fit_the_telegram_64_byte_limit(monkeypatch):
    """Payloads are index ids only — no model name, no action name, well under 64 bytes."""
    choices = [{"value": v, "label": lbl, "is_current": False}
               for v, lbl in GatewayModelCommandsMixin._ROUTING_ACTIONS]
    keyboard, _state = _real_adapter_choice_keyboard(monkeypatch, choices)
    payloads = [b.callback_data for row in keyboard.inline_keyboard for b in row]

    assert payloads == ["cp:0", "cp:1", "cp:2", "cp:3"]
    assert max(len(p.encode("utf-8")) for p in payloads) <= 64
    assert not any("/" in p or "deepseek" in p for p in payloads)


def test_confirm_buttons_also_fit_the_payload_limit(monkeypatch):
    choices = [{"value": v, "label": lbl, "is_current": False}
               for v, lbl, _cur in GatewayModelCommandsMixin._ROUTING_CONFIRM_ACTIONS]
    keyboard, _state = _real_adapter_choice_keyboard(monkeypatch, choices)
    payloads = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert payloads == ["cp:0", "cp:1"]
    assert all(len(p.encode("utf-8")) <= 64 for p in payloads)


def test_enable_button_previews_and_writes_nothing(cfgfile):
    """The Enable tap only opens the confirm step — the preview is the second message."""
    before = cfgfile.read_text()
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    on_action = adapter.calls[0]["on_choice_selected"]

    reply = asyncio.run(on_action("67890", "enable"))

    assert "confirm" in reply.lower()
    assert cfgfile.read_text() == before           # still nothing written
    assert len(adapter.calls) == 2                 # confirm picker sent
    confirm = adapter.calls[1]
    assert [c["value"] for c in confirm["choices"]] == ["confirm", "cancel"]
    assert "enabled: true" in confirm["title"]     # preview precedes the confirm button


def test_confirm_button_enables(cfgfile):
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    asyncio.run(adapter.calls[0]["on_choice_selected"]("67890", "enable"))
    on_confirm = adapter.calls[1]["on_choice_selected"]

    reply = asyncio.run(on_confirm("67890", "confirm"))

    assert "Routing enabled" in reply
    assert "enabled: true" in cfgfile.read_text()


def test_cancel_buttons_write_nothing(cfgfile):
    before = cfgfile.read_text()
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    on_action = adapter.calls[0]["on_choice_selected"]

    assert "Closed" in asyncio.run(on_action("67890", "cancel"))
    assert cfgfile.read_text() == before

    asyncio.run(on_action("67890", "enable"))
    on_confirm = adapter.calls[-1]["on_choice_selected"]
    assert "Cancelled" in asyncio.run(on_confirm("67890", "cancel"))
    assert cfgfile.read_text() == before


def test_change_models_button_is_a_no_op(cfgfile):
    before = cfgfile.read_text()
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    reply = asyncio.run(adapter.calls[0]["on_choice_selected"]("67890", "change_models"))

    assert "/model" in reply
    assert cfgfile.read_text() == before


def test_disable_button_disables(cfgfile):
    _run("on confirm")
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    reply = asyncio.run(adapter.calls[0]["on_choice_selected"]("67890", "disable"))

    assert "Routing disabled" in reply
    assert "enabled: false" in cfgfile.read_text()


def test_unknown_callback_value_writes_nothing(cfgfile):
    """A stale/unknown payload resolves to no action and no write."""
    before = cfgfile.read_text()
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    on_action = adapter.calls[0]["on_choice_selected"]

    for value in ("cp:9", "frobnicate", ""):
        reply = asyncio.run(on_action("67890", value))
        assert "Closed" in reply
    assert cfgfile.read_text() == before
    assert len(adapter.calls) == 1  # no follow-up picker opened


def test_adapter_drops_expired_and_out_of_range_payloads(monkeypatch):
    """The adapter answers and returns before the handler — so nothing can be written."""
    import plugins.platforms.telegram.adapter as tg

    called: list = []

    async def _cb(_chat_id, value):
        called.append(value)
        return ""

    class _Query:
        def __init__(self):
            self.answers: list = []

        async def answer(self, **kwargs):
            self.answers.append(kwargs)

    # Expired: the chat has no live picker state at all.
    query = _Query()
    asyncio.run(tg.TelegramAdapter._handle_choice_picker_callback(
        SimpleNamespace(_choice_picker_state={}), query, "cp:0", "67890"))
    assert called == [] and query.answers

    # Stale: state exists but the payload's index is out of range.
    state = {"choices": [{"value": "enable"}], "on_choice_selected": _cb}
    query = _Query()
    asyncio.run(tg.TelegramAdapter._handle_choice_picker_callback(
        SimpleNamespace(_choice_picker_state={"67890": state}), query, "cp:99", "67890"))
    assert called == [] and query.answers


def test_unauthorized_button_press_writes_nothing(cfgfile):
    """Callbacks re-check the same authorization the command uses (fail closed)."""
    before = cfgfile.read_text()
    adapter = _PickerAdapter()
    runner = _ButtonRunner(adapter)
    asyncio.run(runner._handle_model_routing_command(_PickerEvent("")))
    on_action = adapter.calls[0]["on_choice_selected"]

    runner._authorized = False
    assert "not authorized" in asyncio.run(on_action("67890", "enable"))
    assert "not authorized" in asyncio.run(on_action("67890", "disable"))
    assert cfgfile.read_text() == before


def test_followup_picker_survives_the_first_tap(cfgfile):
    """The adapter keeps the state a follow-up picker registered (confirm step must stay live)."""
    import plugins.platforms.telegram.adapter as tg

    class _Query:
        async def answer(self, **_kwargs):
            return None

    fake = SimpleNamespace(_choice_picker_state={})

    async def _edit(_query, _text):
        return None

    async def _cb(_chat_id, _value):
        # The handler opens the confirm picker mid-callback, registering fresh state.
        fake._choice_picker_state["67890"] = {"choices": [], "on_choice_selected": None}
        return "preview"

    fake._edit_result_text = _edit
    fake._choice_picker_state["67890"] = {"choices": [{"value": "enable"}], "on_choice_selected": _cb}

    asyncio.run(tg.TelegramAdapter._handle_choice_picker_callback(fake, _Query(), "cp:0", "67890"))

    assert "67890" in fake._choice_picker_state  # consumed state did not take the follow-up down
