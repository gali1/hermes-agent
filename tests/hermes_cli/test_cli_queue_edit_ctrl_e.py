"""Ctrl+E composer editing of queued prompts: load, replace, delete, cancel.

While ``_queue_edit_index`` is set, Enter must SAVE the edited text back into
``_pending_input`` through ``_queue_edit`` — never submit it to the model. The
handlers are exercised directly on a ``HermesCLI.__new__`` skeleton with a real
prompt_toolkit ``Buffer`` and a real ``queue.Queue`` (mirrors
tests/hermes_cli/test_tui_rapid_enter_paste.py).
"""

from __future__ import annotations

from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prompt_toolkit.buffer import Buffer

from agent.i18n import t
from cli import HermesCLI


def _shell():
    shell = object.__new__(HermesCLI)
    shell._tui_enter_overlay = lambda event: False
    shell._tui_multiline_shortcuts = False
    shell._tui_last_text_change = 0.0
    shell._attached_images = []
    shell._agent_running = True
    shell._tui_enter_while_busy = MagicMock()
    shell._tui_enter_inline_command = MagicMock(return_value=False)
    shell._inline_pastes = lambda buf: None
    shell._pending_input = Queue()
    shell._queue_edit_index = None
    return shell


def _event(buf: Buffer):
    return SimpleNamespace(
        app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=False))


def _queued(shell) -> list:
    with shell._pending_input.mutex:
        return list(shell._pending_input.queue)


def _ctrl_e(shell, buf: Buffer):
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))


def _enter(shell, buf: Buffer):
    shell._tui_last_text_change = 0.0  # human-scale pause: this Enter is a submit, not a paste
    with patch("cli._cprint") as cprint:
        shell._tui_handle_enter(_event(buf))
    return cprint


def test_ctrl_e_on_empty_queue_prints_and_keeps_edit_mode_off():
    shell = _shell()
    buf = Buffer()
    with patch("cli._cprint") as cprint:
        shell._tui_handle_ctrl_e(_event(buf))
    assert shell._queue_edit_index is None
    assert buf.text == ""
    cprint.assert_called_once_with(f"  {t('cli.queue.edit_empty')}")


def test_ctrl_e_loads_first_queued_prompt_into_the_composer():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._pending_input.put("beta")
    buf = Buffer()
    with patch("cli._cprint") as cprint:
        shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == "alpha"
    assert buf.cursor_position == len("alpha")
    assert shell._queue_edit_index == 1
    cprint.assert_called_once_with(f"  {t('cli.queue.edit_hint', index=1, count=2)}")


def test_ctrl_e_cycles_through_the_queue_and_wraps():
    shell = _shell()
    shell._pending_input.put("a")
    shell._pending_input.put("b")
    buf = Buffer()
    _ctrl_e(shell, buf)
    assert (shell._queue_edit_index, buf.text) == (1, "a")
    _ctrl_e(shell, buf)
    assert (shell._queue_edit_index, buf.text) == (2, "b")
    _ctrl_e(shell, buf)
    assert (shell._queue_edit_index, buf.text) == (1, "a")


def test_ctrl_e_restarts_at_one_when_the_index_is_out_of_range():
    shell = _shell()
    shell._pending_input.put("a")
    shell._pending_input.put("b")
    shell._queue_edit_index = 9
    buf = Buffer()
    _ctrl_e(shell, buf)
    assert (shell._queue_edit_index, buf.text) == (1, "a")


def test_ctrl_e_is_ignored_while_a_modal_owns_the_composer():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._sudo_state = {"response_queue": Queue()}
    buf = Buffer()
    shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == ""
    assert shell._queue_edit_index is None


def test_enter_saves_into_the_queue_and_never_submits_to_the_model():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._pending_input.put("beta")
    buf = Buffer()
    _ctrl_e(shell, buf)
    buf.text = "alpha edited"
    _enter(shell, buf)
    assert _queued(shell) == ["alpha edited", "beta"]
    assert shell._queue_edit_index is None
    assert buf.text == ""
    shell._tui_enter_while_busy.assert_not_called()
    shell._tui_enter_inline_command.assert_not_called()


def test_enter_with_a_blank_composer_is_rejected_and_the_queue_is_untouched():
    shell = _shell()
    shell._pending_input.put("alpha")
    buf = Buffer()
    _ctrl_e(shell, buf)
    buf.text = ""
    cprint = _enter(shell, buf)
    assert _queued(shell) == ["alpha"]
    assert shell._queue_edit_index == 1  # still editing, so the user can keep typing
    cprint.assert_called_once_with(f"  {t('cli.queue.edit_blank')}")
    shell._tui_enter_while_busy.assert_not_called()


def test_escape_cancels_the_edit_and_keeps_the_queued_prompt():
    shell = _shell()
    shell._pending_input.put("alpha")
    buf = Buffer()
    _ctrl_e(shell, buf)
    buf.text = "changed"
    with patch("cli._cprint") as cprint:
        shell._tui_handle_queue_edit_cancel(_event(buf))
    assert _queued(shell) == ["alpha"]
    assert shell._queue_edit_index is None
    assert buf.text == ""
    cprint.assert_called_once_with(f"  {t('cli.queue.edit_cancelled')}")


def test_ctrl_x_deletes_the_edited_item_and_preserves_the_rest():
    shell = _shell()
    for item in ("a", "b", "c"):
        shell._pending_input.put(item)
    buf = Buffer()
    _ctrl_e(shell, buf)
    _ctrl_e(shell, buf)  # editing "b"
    assert shell._queue_edit_index == 2
    with patch("cli._cprint") as cprint:
        shell._tui_handle_queue_edit_delete(_event(buf))
    assert _queued(shell) == ["a", "c"]
    assert shell._queue_edit_index is None
    assert buf.text == ""
    cprint.assert_any_call(f"  {t('cli.queue.removed', index=2, preview='b')}")


def test_save_after_the_queue_emptied_mid_edit_reports_not_found_and_clears_state():
    shell = _shell()
    shell._pending_input.put("alpha")
    buf = Buffer()
    _ctrl_e(shell, buf)
    shell._pending_input.get_nowait()  # the loop consumed it while the user was editing
    buf.text = "alpha edited"
    cprint = _enter(shell, buf)
    assert _queued(shell) == []
    assert shell._queue_edit_index is None
    assert buf.text == ""
    cprint.assert_any_call(f"  {t('cli.queue.item_not_found', index=1, size=0)}")
    shell._tui_enter_while_busy.assert_not_called()


def test_save_after_the_edited_item_was_consumed_leaves_remaining_queue_untouched():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._pending_input.put("beta")
    buf = Buffer()
    _ctrl_e(shell, buf)
    _ctrl_e(shell, buf)  # editing index 2 ("beta")
    shell._pending_input.get_nowait()  # "alpha" consumed; index 2 no longer exists
    buf.text = "beta edited"
    _enter(shell, buf)
    assert _queued(shell) == ["beta"]
    assert shell._queue_edit_index is None
    assert buf.text == ""


def test_multi_item_edit_leaves_other_items_and_their_order_intact():
    shell = _shell()
    for item in ("one", "two", "three"):
        shell._pending_input.put(item)
    buf = Buffer()
    _ctrl_e(shell, buf)
    _ctrl_e(shell, buf)
    buf.text = "TWO"
    _enter(shell, buf)
    assert _queued(shell) == ["one", "TWO", "three"]


def test_special_and_multiline_content_round_trips():
    shell = _shell()
    shell._pending_input.put("line one\nline two")
    buf = Buffer()
    _ctrl_e(shell, buf)
    # The shared dock extractor collapses whitespace for its one-line preview.
    assert buf.text == "line one line two"
    edited = 'line one\nline "two"\n$HOME [not a paste]'
    buf.text = edited
    _enter(shell, buf)
    assert _queued(shell) == [edited]


def test_tuple_item_is_loaded_and_replaced_with_plain_text():
    shell = _shell()
    shell._pending_input.put(("look at this", ["shot.png"]))
    buf = Buffer()
    _ctrl_e(shell, buf)
    assert buf.text == "look at this"
    buf.text = "look at that"
    _enter(shell, buf)
    assert _queued(shell) == ["look at that"]


def test_key_bindings_register_ctrl_e_escape_and_ctrl_x_with_escape_last():
    """The queue-edit bindings are registered after the generic Escape handlers, and the
    filtered Escape binding is eager — otherwise double-Escape/Alt-sequences would win."""
    shell = _shell()
    shell.config = {}
    shell._prompt_stash = MagicMock()
    kb = shell._tui_build_key_bindings()

    def handler_name(binding):
        handler = binding.handler
        return getattr(handler, "__func__", handler).__name__

    handlers = [(binding.keys, handler_name(binding)) for binding in kb.bindings]
    assert (("c-e",), "_tui_handle_ctrl_e") in handlers
    assert (("c-x",), "_tui_handle_queue_edit_delete") in handlers
    escapes = [binding for binding in kb.bindings if binding.keys == ("escape",)]
    assert handler_name(escapes[-1]) == "_tui_handle_queue_edit_cancel"
    assert escapes[-1].eager()


def test_voice_item_edit_keeps_the_voice_sentinel():
    # Imported here (not at module scope): another test file reloads ``cli``, and the
    # sentinel class must be the one the live module's ``_queue_edit`` resolves.
    from cli import _VoiceInputMessage
    shell = _shell()
    shell._pending_input.put(_VoiceInputMessage("spoken ask"))
    buf = Buffer()
    _ctrl_e(shell, buf)
    assert buf.text == "spoken ask"
    buf.text = "spoken ask, refined"
    _enter(shell, buf)
    items = _queued(shell)
    assert len(items) == 1
    assert isinstance(items[0], _VoiceInputMessage)
    assert items[0].text == "spoken ask, refined"
