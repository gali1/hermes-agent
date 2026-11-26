"""Ctrl+E query editing: numbered queued-prompt picker, last-submitted recall, real keystrokes.

Three things this file exists to prove, in order of importance:

1. **The keystroke reaches the handler at runtime.** Every earlier test inspected
   ``kb.bindings`` for a ``("c-e",)`` entry. That proves registration, not dispatch:
   prompt_toolkit calls the *last* matching binding, so a later ``c-e`` (the default
   emacs ``end-of-line``) would silently win while the registration test stayed green.
   ``test_real_ctrl_e_keystroke_*`` feeds ``\\x05`` through a real ``Application`` with
   a real ``PipeInput`` and asserts the composer changed.
2. **The empty-queue case is no longer a dead key.** It restores the most recently
   submitted prompt from the CLI history file.
3. **Queued prompts are individually selectable** — the menu prints ``Ctrl+E N`` next
   to each item and the matching digit loads exactly that item.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory

from agent.i18n import t
from cli import HermesCLI


# ── harness ──────────────────────────────────────────────────────────────────
class _Skeleton(HermesCLI):
    """``HermesCLI`` without ``__init__``, tolerant of attributes other mixins own.

    A real keystroke makes prompt_toolkit evaluate the filter of *every* binding that
    matches the key, not just the one under test. Those filters read state owned by
    mixins this skeleton never initialised (``_clarify_state``, ``_model_picker_state``,
    …) and would abort the event loop. Missing attributes therefore read as ``None``
    here, which is the "overlay not open" state those filters are asking about.
    """

    def __getattr__(self, name):          # only called when normal lookup fails
        if name.startswith("__"):
            raise AttributeError(name)
        return self.__dict__.setdefault(name, None)


def _shell(history_file: Path | None = None):
    shell = object.__new__(_Skeleton)
    shell._tui_enter_overlay = lambda event: False
    shell._tui_multiline_shortcuts = False
    shell._tui_last_text_change = 0.0
    shell._attached_images = []
    shell._agent_running = True
    shell._tui_enter_while_busy = MagicMock()
    shell._tui_enter_inline_command = MagicMock(return_value=False)
    shell._inline_pastes = lambda buf: None
    shell._pending_input = Queue()
    shell.config = {}
    shell._prompt_stash = MagicMock()
    shell._queue_edit_index = None
    shell._queue_edit_picker = False
    shell._queue_edit_history = False
    if history_file is not None:
        shell._history_file = history_file
    return shell


def _event(buf: Buffer):
    return SimpleNamespace(
        app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=False))


def _queued(shell) -> list:
    with shell._pending_input.mutex:
        return list(shell._pending_input.queue)


def _write_history(path: Path, entries: list[str]) -> None:
    """Persist entries the way prompt_toolkit's FileHistory does."""
    hist = FileHistory(str(path))
    for entry in entries:
        hist.store_string(entry)


def _press_digit(shell, buf: Buffer, number: int):
    with patch("cli._cprint") as cprint:
        shell._queue_edit_digit_handler(number)(_event(buf))
    return cprint


# ── queued-prompt picker ─────────────────────────────────────────────────────
def test_ctrl_e_lists_every_queued_prompt_with_its_shortcut():
    shell = _shell()
    for text in ("alpha", "beta", "gamma"):
        shell._pending_input.put(text)
    buf = Buffer()
    with patch("cli._cprint") as cprint:
        shell._tui_handle_ctrl_e(_event(buf))

    printed = "\n".join(str(c.args[0]) for c in cprint.call_args_list)
    assert "Ctrl+E 1" in printed and "Ctrl+E 2" in printed and "Ctrl+E 3" in printed
    assert "alpha" in printed and "beta" in printed and "gamma" in printed
    assert t("cli.queue.edit_menu_header", count=3) in printed

    assert buf.text == "alpha"                       # first item is loaded
    assert shell._queue_edit_index == 1
    assert shell._queue_edit_picker is True           # digits armed


def test_digit_selects_the_matching_queued_prompt():
    shell = _shell()
    for text in ("alpha", "beta", "gamma"):
        shell._pending_input.put(text)
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
    _press_digit(shell, buf, 2)

    assert buf.text == "beta"
    assert shell._queue_edit_index == 2
    assert shell._queue_edit_picker is False          # selection closes the menu


def test_invalid_digit_explains_range_and_keeps_composer():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._pending_input.put("beta")
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
    cprint = _press_digit(shell, buf, 7)

    assert cprint.call_args.args[0] == f"  {t('cli.queue.edit_select_invalid', index=7, count=2)}"
    assert buf.text == "alpha"                        # selection refused, nothing lost
    assert shell._queue_edit_picker is False


def test_repeated_ctrl_e_cycles_through_the_queue_and_wraps():
    shell = _shell()
    for text in ("alpha", "beta"):
        shell._pending_input.put(text)
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "alpha"
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "beta"
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "alpha"                    # wrapped


def test_numbering_tracks_live_queue_order_after_a_delete():
    shell = _shell()
    for text in ("alpha", "beta", "gamma"):
        shell._pending_input.put(text)
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))         # loads alpha (index 1)
        shell._queue_remove("1")                      # beta is now index 1, gamma index 2
        shell._tui_handle_queue_edit_cancel(_event(buf))
        shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == "beta"                         # numbering re-read from the live queue


# ── last-submitted recall ────────────────────────────────────────────────────
def test_empty_queue_restores_the_last_submitted_prompt(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["first query", "second query"])
    shell = _shell(hist)
    buf = Buffer()
    with patch("cli._cprint") as cprint:
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == "second query"
    assert buf.cursor_position == len("second query")
    assert shell._queue_edit_index is None       # Enter must submit, not save back
    assert shell._queue_edit_history is True
    assert cprint.call_args.args[0] == f"  {t('cli.queue.edit_history_restored')}"


def test_restored_history_prompt_is_editable_and_submits_normally(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["original ask"])
    shell = _shell(hist)
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == "original ask"

    buf.text = "original ask, refined"
    consumed = shell._tui_handle_queue_edit_enter(_event(buf))
    assert consumed is False                       # falls through to the normal submit path
    assert shell._queue_edit_history is False      # flag cleared for the next turn
    assert _queued(shell) == []                    # nothing was written into the queue


def test_slash_commands_are_skipped_when_recalling_history(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["a real question", "/model"])
    shell = _shell(hist)
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == "a real question"


def test_queue_wins_over_history_when_both_exist(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["submitted earlier"])
    shell = _shell(hist)
    shell._pending_input.put("still queued")
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == "still queued"
    assert shell._queue_edit_index == 1


def test_multiline_and_unicode_survive_the_round_trip():
    shell = _shell()
    text = "line one\nline two — ünïcode ✓"
    shell._pending_input.put(text)
    buf = Buffer()
    with patch("cli._cprint"):
        shell._tui_handle_ctrl_e(_event(buf))
    assert buf.text == text


def test_escape_cancels_and_restores_nothing():
    shell = _shell()
    shell._pending_input.put("alpha")
    buf = Buffer()
    with patch("cli._cprint") as cprint:
        shell._tui_handle_ctrl_e(_event(buf))
        shell._tui_handle_queue_edit_cancel(_event(buf))
    assert buf.text == ""
    assert shell._queue_edit_index is None
    assert shell._queue_edit_picker is False
    assert shell._queue_edit_history is False
    assert _queued(shell) == ["alpha"]             # queue untouched
    assert cprint.call_args.args[0] == f"  {t('cli.queue.edit_cancelled')}"


# ── real keystroke dispatch ──────────────────────────────────────────────────
def _run_with_keystrokes(shell, buf, keystrokes: str, deadline_s: float = 5.0,
                         ready=None) -> None:
    """Drive a real prompt_toolkit Application with real bytes from a pseudo terminal."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import BufferControl, HSplit, Layout, Window
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        app = Application(
            layout=Layout(HSplit([Window(BufferControl(buffer=buf))])),
            key_bindings=shell._tui_build_key_bindings(),
            input=pipe,
            output=DummyOutput(),
            full_screen=False,
        )
        thread = threading.Thread(target=app.run, daemon=True)
        thread.start()
        time.sleep(0.15)                       # let the event loop attach to the pipe
        pipe.send_text(keystrokes)

        ready = ready or (lambda: bool(buf.text))
        deadline = time.time() + deadline_s
        while time.time() < deadline and not ready():
            time.sleep(0.02)
        app.exit()
        thread.join(2)


@pytest.mark.parametrize("keystrokes", ["\x05"])
def test_real_ctrl_e_keystroke_loads_the_queued_prompt(keystrokes):
    """\\x05 through a real input pipe must run the editor, not emacs end-of-line.

    With an empty queue this asserts on the restored-history path so the test does not
    depend on filesystem history: the buffer must change either way. A binding that is
    shadowed by prompt_toolkit's default ``c-e`` leaves the buffer empty here.
    """
    shell = _shell()
    shell._pending_input.put("queued via keystroke")
    buf = Buffer()
    with patch("cli._cprint"):
        _run_with_keystrokes(shell, buf, keystrokes)
    assert buf.text == "queued via keystroke", (
        "Ctrl+E did not reach the composer editor — a later c-e binding (emacs "
        "end-of-line) is winning the key"
    )
    assert shell._queue_edit_index == 1


def test_real_ctrl_e_then_digit_selects_the_second_item():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._pending_input.put("beta")
    buf = Buffer()
    with patch("cli._cprint"):
        _run_with_keystrokes(shell, buf, "\x052",    # Ctrl+E, then "2"
                             ready=lambda: shell._queue_edit_index == 2)
    assert buf.text == "beta"
    assert shell._queue_edit_index == 2


def test_real_ctrl_e_with_empty_queue_and_no_history_still_answers():
    """The original bug: nothing at all happened. Now the key always produces output."""
    shell = _shell()
    shell._history_file = Path("/nonexistent/hermes-history-test")
    buf = Buffer()
    with patch("cli._cprint") as cprint:
        _run_with_keystrokes(shell, buf, "\x05")
    printed = " ".join(str(c.args[0]) for c in cprint.call_args_list)
    assert t("cli.queue.edit_nothing") in printed
