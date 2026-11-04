"""`/model-routing` inline buttons (Section 6.3): entry actions, model multi-select, confirm.

Two layers are covered:

* the Telegram adapter's generic multi-select picker (toggle, paging, All/None, Done,
  Cancel) — including Telegram's 64-byte ``callback_data`` cap, which is met by sending
  an index and keeping the model names server-side;
* the runner's button surface, which must reach the same appliers and the same
  ``update_routing`` writer as the typed flow, and must fall back to the numbered text
  flow on any platform (or after any send failure) instead of dropping the command.

Every write goes to a scratch ``HERMES_HOME`` (``tmp_path``); the real ``~/.hermes`` is
never touched (``tests/home_io_guard.py`` enforces it).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent import model_router as mr
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter

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


def _event(text: str, *, user_id: str = "1", chat_id: str = "2") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id=user_id, chat_id=chat_id, user_name="tester",
    )
    return MessageEvent(text=text, source=source)


class _ButtonAdapter:
    """Minimal adapter type exposing both pickers (the capability gate the runner checks)."""

    def __init__(self, *, choices_ok: bool = True, multi_ok: bool = True):
        self.choice_calls: list = []
        self.multi_calls: list = []
        self._choices_ok = choices_ok
        self._multi_ok = multi_ok

    async def send_choice_picker(self, **kwargs):
        self.choice_calls.append(kwargs)
        return SendResult(success=self._choices_ok, message_id="m1")

    async def send_multi_select_picker(self, **kwargs):
        self.multi_calls.append(kwargs)
        return SendResult(success=self._multi_ok, message_id="m2")


class _ChoiceOnlyAdapter:
    """Adapter with a single-choice picker but no multi-select (e.g. a text-ish platform)."""

    def __init__(self):
        self.choice_calls: list = []

    async def send_choice_picker(self, **kwargs):
        self.choice_calls.append(kwargs)
        return SendResult(success=True, message_id="m1")


class _TextOnlyAdapter:
    """Adapter with no picker methods at all — the text-only platform shape."""


def _make_runner(adapter=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._delivery_adapter_for = lambda source: adapter
    runner._reply_metadata = lambda event: {}
    return runner


# ─────────────────────────── adapter: multi-select ───────────────────────────


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))
    adapter._app = MagicMock()
    return adapter


def _query(data: str = "") -> AsyncMock:
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.from_user = MagicMock()
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


def _options(*names: str) -> list:
    return [{"value": n, "label": n} for n in names]


@pytest.fixture
def button_types(monkeypatch):
    """Deterministic stand-ins for PTB's keyboard types.

    The suite runs against a mocked ``telegram`` module, so a real ``InlineKeyboardMarkup``
    is a MagicMock and its rows cannot be inspected. These stand-ins keep the shape the
    adapter uses (``.inline_keyboard`` rows of objects with ``.text``/``.callback_data``)
    so labels, payload sizes and paging are actually asserted.
    """
    import plugins.platforms.telegram.adapter as adapter_mod

    class Button:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data

    class Markup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(adapter_mod, "InlineKeyboardButton", Button)
    monkeypatch.setattr(adapter_mod, "InlineKeyboardMarkup", Markup)
    return Button, Markup


class TestMultiSelectPicker:
    @pytest.mark.asyncio
    async def test_sends_one_button_per_option_plus_controls(self, button_types):
        adapter = _make_adapter()
        result = await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick models", options=_options("a/one", "b/two"),
            selected=["b/two"], session_key="s", on_selection_done=AsyncMock(),
        )
        assert result.success is True
        sent = adapter._bot.send_message.call_args.kwargs
        keyboard = sent["reply_markup"]
        labels = [b.text for row in keyboard.inline_keyboard for b in row]
        assert any("one" in label for label in labels)
        assert any(label.startswith("✓") and "two" in label for label in labels)
        assert "✅ Done" in labels and "✖ Cancel" in labels
        assert "All" in labels and "None" in labels

    @pytest.mark.asyncio
    async def test_callback_data_stays_within_telegram_limit(self, button_types):
        """The 64-byte cap is met by carrying an index, never a model name."""
        adapter = _make_adapter()
        long_names = [f"vendor/{'x' * 60}-{i}" for i in range(12)]
        await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick", options=_options(*long_names),
            selected=[], session_key="s", on_selection_done=AsyncMock(),
        )
        keyboard = adapter._bot.send_message.call_args.kwargs["reply_markup"]
        payloads = [b.callback_data for row in keyboard.inline_keyboard for b in row]
        assert payloads
        assert all(len(p.encode()) <= 64 for p in payloads)
        assert all("vendor" not in p for p in payloads)

    @pytest.mark.asyncio
    async def test_toggle_marks_and_done_returns_values_in_option_order(self):
        adapter = _make_adapter()
        done = AsyncMock(return_value="saved")
        await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick", options=_options("m1", "m2", "m3"),
            selected=[], session_key="s", on_selection_done=done,
        )
        # taps arrive out of order: m3 then m1
        await adapter._handle_multi_select_callback(_query("mrs:2"), "mrs:2", "12345")
        await adapter._handle_multi_select_callback(_query("mrs:0"), "mrs:0", "12345")
        query = _query("mrsd")
        await adapter._handle_multi_select_callback(query, "mrsd", "12345")

        done.assert_awaited_once()
        assert done.await_args.args[1] == ["m1", "m3"]  # option order, not tap order
        # the picker is consumed and the final text replaces it
        assert "12345" not in adapter._multi_select_state
        assert query.edit_message_text.await_count == 1

    @pytest.mark.asyncio
    async def test_toggle_re_renders_with_check_mark(self, button_types):
        adapter = _make_adapter()
        await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick", options=_options("m1", "m2"),
            selected=[], session_key="s", on_selection_done=AsyncMock(),
        )
        query = _query("mrs:1")
        await adapter._handle_multi_select_callback(query, "mrs:1", "12345")
        keyboard = query.edit_message_text.call_args.kwargs["reply_markup"]
        labels = [b.text for row in keyboard.inline_keyboard for b in row]
        assert any(label.startswith("✓") and "m2" in label for label in labels)
        assert not any(label.startswith("✓") and "m1" in label for label in labels)

    @pytest.mark.asyncio
    async def test_paging_keeps_selection_and_toggles_the_right_index(self, button_types):
        adapter = _make_adapter()
        done = AsyncMock(return_value="ok")
        names = [f"m{i}" for i in range(10)]
        await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick", options=_options(*names),
            selected=[], session_key="s", on_selection_done=done, page_size=8,
        )
        await adapter._handle_multi_select_callback(_query("mrsv:1"), "mrsv:1", "12345")
        state = adapter._multi_select_state["12345"]
        assert state["page"] == 1
        await adapter._handle_multi_select_callback(_query("mrs:9"), "mrs:9", "12345")
        await adapter._handle_multi_select_callback(_query("mrsd"), "mrsd", "12345")
        assert done.await_args.args[1] == ["m9"]

    @pytest.mark.asyncio
    async def test_cancel_applies_nothing(self):
        adapter = _make_adapter()
        done = AsyncMock(return_value="ok")
        await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick", options=_options("m1", "m2"),
            selected=["m1"], session_key="s", on_selection_done=done,
        )
        query = _query("mrsx")
        await adapter._handle_multi_select_callback(query, "mrsx", "12345")
        done.assert_not_awaited()
        assert "12345" not in adapter._multi_select_state

    @pytest.mark.asyncio
    async def test_expired_state_answers_instead_of_crashing(self):
        adapter = _make_adapter()
        query = _query("mrsd")
        await adapter._handle_multi_select_callback(query, "mrsd", "999")
        assert query.answer.await_count == 1
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_reaches_the_handler_from_the_callback_prefix(self):
        """The prefix table in ``_handle_callback_query`` is the thing that makes a tap work."""
        import os
        from unittest.mock import patch

        adapter = _make_adapter()
        done = AsyncMock(return_value="saved")
        await adapter.send_multi_select_picker(
            chat_id="12345", title="Pick", options=_options("m1"), selected=[],
            session_key="s", on_selection_done=done,
        )
        query = _query("mrsd")
        update = MagicMock()
        update.callback_query = query
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        done.assert_awaited_once()
        assert "12345" not in adapter._multi_select_state


# ─────────────────────────── runner: button surface ───────────────────────────


class TestRoutingEntryButtons:
    @pytest.mark.asyncio
    async def test_bare_command_sends_the_six_actions(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_model_routing_command(_event("/model-routing"))

        assert result is None  # the adapter owns the reply
        assert len(adapter.choice_calls) == 1
        values = [c["value"] for c in adapter.choice_calls[0]["choices"]]
        assert values == ["1", "2", "3", "4", "5", "6"]
        assert "🧭 Model routing" in adapter.choice_calls[0]["title"]

    @pytest.mark.asyncio
    async def test_text_only_platform_keeps_the_numbered_menu(self, routing_home):
        runner = _make_runner(_TextOnlyAdapter())

        result = await runner._handle_model_routing_command(_event("/model-routing"))

        assert isinstance(result, str)
        assert "Reply /model-routing <number>" in result
        assert "1 Enable routing" in result


    @pytest.mark.asyncio
    async def test_abandoned_flow_is_announced_in_the_button_title(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        event = _event("/model-routing")
        key = runner._model_routing_flow_key(event)
        await runner._handle_model_routing_command(event)  # opens the picker + a flow
        runner._model_routing_flows()[key]["ts"] = 0.0  # long abandoned
        runner._model_routing_flows()[key]["changes"] = {"enabled": True}

        result = await runner._handle_model_routing_command(event)

        assert result is None
        assert "timed out" in adapter.choice_calls[-1]["title"]
        assert mr.routing_settings().get("enabled") is not True  # the pending change died with it


class TestRoutingButtonsFlow:
    @pytest.mark.asyncio
    async def test_enable_asks_before_writing_then_saves(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        event = _event("/model-routing")
        original = (routing_home / "config.yaml").read_bytes()

        await runner._handle_model_routing_command(event)
        await adapter.choice_calls[0]["on_choice_selected"]("2", "1")  # Enable

        # A confirmation picker, not a write.
        assert (routing_home / "config.yaml").read_bytes() == original
        confirm = adapter.choice_calls[-1]
        assert [c["value"] for c in confirm["choices"]] == ["save", "discard"]
        assert "enabled: off → on" in confirm["title"]

        saved = await confirm["on_choice_selected"]("2", "save")
        assert "Saved" in saved
        assert mr.routing_settings()["enabled"] is True

    @pytest.mark.asyncio
    async def test_discard_leaves_the_config_byte_identical(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        event = _event("/model-routing")
        original = (routing_home / "config.yaml").read_bytes()

        await runner._handle_model_routing_command(event)
        await adapter.choice_calls[0]["on_choice_selected"]("2", "1")
        reply = await adapter.choice_calls[-1]["on_choice_selected"]("2", "discard")

        assert "Cancelled" in reply
        assert (routing_home / "config.yaml").read_bytes() == original

    @pytest.mark.asyncio
    async def test_models_button_opens_multiselect_and_done_writes_priority_order(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        event = _event("/model-routing")

        await runner._handle_model_routing_command(event)
        await adapter.choice_calls[0]["on_choice_selected"]("2", "3")  # Change models

        assert len(adapter.multi_calls) == 1
        multi = adapter.multi_calls[0]
        offered = [o["value"] for o in multi["options"]]
        assert "anthropic/claude-haiku" in offered

        # Done with two models, chosen in the offered order.
        await multi["on_selection_done"]("2", ["openai/gpt-5-nano", "anthropic/claude-haiku"])
        confirm = adapter.choice_calls[-1]
        assert [c["value"] for c in confirm["choices"]] == ["save", "discard"]
        assert "models:" in confirm["title"]

        await confirm["on_choice_selected"]("2", "save")
        settings = mr.routing_settings()
        assert settings["models"] == ["openai/gpt-5-nano", "anthropic/claude-haiku"]
        assert settings["preferred_models_only"] is True

    @pytest.mark.asyncio
    async def test_multiselect_unsupported_falls_back_to_the_numbered_page(self, routing_home):
        adapter = _ChoiceOnlyAdapter()
        runner = _make_runner(adapter)

        await runner._handle_model_routing_command(_event("/model-routing"))
        reply = await adapter.choice_calls[0]["on_choice_selected"]("2", "3")

        assert "Reply /model-routing 1,3" in reply

    @pytest.mark.asyncio
    async def test_preferences_button_hands_over_to_the_typed_step(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        event = _event("/model-routing")

        await runner._handle_model_routing_command(event)
        reply = await adapter.choice_calls[0]["on_choice_selected"]("2", "4")

        assert "Preferences" in reply
        step = await runner._handle_model_routing_command(_event("/model-routing 1"))
        assert "Which strategy" in step
        typed = await runner._handle_model_routing_command(_event("/model-routing cost"))
        assert "strategy" in typed and "cost" in typed
        assert mr.routing_settings().get("enabled") is not True  # nothing written yet

    @pytest.mark.asyncio
    async def test_close_applies_nothing(self, routing_home):
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        original = (routing_home / "config.yaml").read_bytes()

        await runner._handle_model_routing_command(_event("/model-routing"))
        reply = await adapter.choice_calls[0]["on_choice_selected"]("2", "6")

        assert "Cancelled" in reply
        assert (routing_home / "config.yaml").read_bytes() == original

    @pytest.mark.asyncio
    async def test_unauthorized_tap_cannot_write(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(ROUTED_CONFIG, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        adapter = _ButtonAdapter()
        runner = _make_runner(adapter)
        runner._is_user_authorized = lambda source: False
        original = (home / "config.yaml").read_bytes()

        reply = await runner._model_routing_button_action(_event("/model-routing"), "save")

        assert "🔒" in reply
        assert (home / "config.yaml").read_bytes() == original
