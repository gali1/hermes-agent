"""Ctrl+E queued-prompt editing (hermes-agent tree).

The queue used to be append-only: ``/queue <prompt>`` added entries and nothing
could list, change or remove them. These tests pin the replacement behaviour —
a numbered menu, per-item selection, in-place save, delete, cancel, and the
history fallback when the queue is empty — and they drive it through a real
``Buffer`` so the composer is exercised the same way the REPL drives it.
"""

from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from prompt_toolkit.buffer import Buffer

from cli import HermesCLI


class _Skeleton(HermesCLI):
    """``HermesCLI`` without ``__init__``; unknown mixin state reads as None."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.__dict__.setdefault(name, None)


def _shell(history_file: Path | None = None):
    shell = object.__new__(_Skeleton)
    shell._pending_input = queue.Queue()
    shell._interrupt_queue = queue.Queue()
    shell._agent_running = False
    shell._queue_edit_index = None
    shell._queue_edit_picker = False
    shell._queue_edit_history = False
    shell._attached_images = []
    shell._history_file = history_file or Path("/nonexistent/hermes-agent-history")
    return shell


def _event(buf: Buffer):
    return SimpleNamespace(
        current_buffer=buf,
        app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=False),
    )


@contextmanager
def _prints():
    """Collect what the handlers print, without depending on global stdout."""
    with patch("cli._cprint") as cprint:
        yield cprint
        cprint.lines = [str(c.args[0]) for c in cprint.call_args_list if c.args]


def _queued(shell) -> list:
    with shell._pending_input.mutex:
        return list(shell._pending_input.queue)


def _write_history(path: Path, entries: list[str]) -> None:
    path.write_text(
        "".join(f"\n# 2026-01-01 00:00:00.000000\n" + "".join(f"+{l}\n" for l in e.split("\n"))
                for e in entries),
        encoding="utf-8",
    )


# ── behaviour ────────────────────────────────────────────────────────────────
def test_ctrl_e_lists_queue_with_numbered_shortcuts():
    shell = _shell()
    for text in ("alpha", "beta", "gamma"):
        shell._pending_input.put(text)
    buf = Buffer()

    with _prints() as out:
        shell._tui_handle_ctrl_e(_event(buf))
    printed = "\n".join(out.lines)

    assert "Ctrl+E 1" in printed and "Ctrl+E 2" in printed and "Ctrl+E 3" in printed
    assert "alpha" in printed and "beta" in printed and "gamma" in printed
    assert buf.text == "alpha"
    assert shell._queue_edit_index == 1
    assert shell._queue_edit_picker is True


def test_digit_selects_that_queue_entry():
    shell = _shell()
    for text in ("alpha", "beta"):
        shell._pending_input.put(text)
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))
        shell._queue_edit_digit(_event(buf), 2)

    assert buf.text == "beta"
    assert shell._queue_edit_index == 2
    assert shell._queue_edit_picker is False


def test_invalid_number_is_refused_without_losing_the_draft():
    shell = _shell()
    shell._pending_input.put("alpha")
    buf = Buffer()
    with _prints() as out:
        shell._tui_handle_ctrl_e(_event(buf))
        shell._queue_edit_digit(_event(buf), 5)

    assert buf.text == "alpha"
    assert any("No queued prompt 5" in line for line in out.lines)


def test_enter_saves_the_edit_back_into_the_same_slot():
    shell = _shell()
    for text in ("alpha", "beta", "gamma"):
        shell._pending_input.put(text)
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))
        shell._queue_edit_digit(_event(buf), 2)
        buf.text = "beta, revised"
        assert shell._tui_handle_queue_edit_save(_event(buf)) is True

    assert _queued(shell) == ["alpha", "beta, revised", "gamma"]
    assert buf.text == ""
    assert shell._queue_edit_index is None


def test_image_attached_items_keep_their_images_through_an_edit():
    shell = _shell()
    shell._pending_input.put(("look at this", ["shot.png"]))
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "look at this"
        buf.text = "look at that"
        shell._tui_handle_queue_edit_save(_event(buf))

    assert _queued(shell) == [("look at that", ["shot.png"])]


def test_ctrl_x_removes_only_the_edited_entry():
    shell = _shell()
    for text in ("alpha", "beta", "gamma"):
        shell._pending_input.put(text)
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))
        shell._queue_edit_digit(_event(buf), 2)
        shell._tui_handle_queue_edit_delete(_event(buf))

    assert _queued(shell) == ["alpha", "gamma"]
    assert buf.text == ""


def test_escape_cancels_and_leaves_the_queue_untouched():
    shell = _shell()
    shell._pending_input.put("alpha")
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))
        buf.text = "half-edited"
        shell._tui_handle_queue_edit_cancel(_event(buf))

    assert buf.text == ""
    assert _queued(shell) == ["alpha"]
    assert shell._queue_edit_index is None


def test_repeated_ctrl_e_cycles_and_wraps():
    shell = _shell()
    for text in ("alpha", "beta"):
        shell._pending_input.put(text)
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "alpha"
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "beta"
        shell._tui_handle_ctrl_e(_event(buf))
        assert buf.text == "alpha"


def test_multiline_and_unicode_are_preserved():
    shell = _shell()
    text = "line one\nline two — ünïcode ✓"
    shell._pending_input.put(text)
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == text


def test_empty_queue_restores_the_last_submitted_prompt(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["first ask", "second ask"])
    shell = _shell(hist)
    buf = Buffer()

    with _prints() as out:
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == "second ask"
    assert shell._queue_edit_index is None      # Enter submits, it never saves back
    assert shell._queue_edit_history is True
    assert any("Restored your last submitted prompt" in line for line in out.lines)


def test_slash_commands_are_skipped_when_recalling(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["a real question", "/model"])
    shell = _shell(hist)
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == "a real question"


def test_queue_wins_over_history(tmp_path):
    hist = tmp_path / "history"
    _write_history(hist, ["submitted earlier"])
    shell = _shell(hist)
    shell._pending_input.put("still queued")
    buf = Buffer()
    with _prints():
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == "still queued"
    assert shell._queue_edit_index == 1


def test_nothing_to_edit_says_so():
    shell = _shell()
    buf = Buffer()
    with _prints() as out:
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == ""
    assert any("Nothing to edit yet" in line for line in out.lines)


def test_ctrl_e_explains_itself_when_a_modal_prompt_is_open():
    shell = _shell()
    shell._pending_input.put("alpha")
    shell._approval_state = {"options": []}      # an approval prompt owns the keyboard
    buf = Buffer()

    with _prints() as out:
        shell._tui_handle_ctrl_e(_event(buf))

    assert buf.text == ""
    assert any("waiting on the open prompt" in line for line in out.lines)


@pytest.mark.parametrize("number", [1, 2, 3, 9])
def test_every_digit_below_ten_is_bound(number):
    """The menu advertises Ctrl+E 1..9, so the builder must register all of them.

    This tree builds its KeyBindings inline in ``run()`` rather than in a reusable
    method, so the assertion is made against the builder's own source: the loop is
    what registers the digits, and it must cover 1..9.
    """
    text = _cli_source()
    assert "for _qe_num in range(1, 10):" in text
    assert "kb.add(str(_qe_num), filter=_qe_picker)" in text


def _cli_source() -> str:
    return (Path(__file__).resolve().parents[2] / "cli.py").read_text(encoding="utf-8")


def test_bindings_are_declared_in_the_tui_builder():
    """Every gesture the menu advertises exists as a binding, built in cli.py."""
    text = _cli_source()
    for gesture in ("@kb.add('c-e')", "@kb.add('enter', filter=_qe_edit)",
                    "@kb.add('c-x', filter=_qe_edit)",
                    "@kb.add('escape', filter=_qe_session, eager=True)"):
        assert gesture in text, gesture
