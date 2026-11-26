"""Composer editing gestures for the interactive CLI input area.

Topical sibling of ``cli_tui_mixin``: this mixin owns the Ctrl+E editing session
(``_queue_edit_index`` — load from ``_pending_input``, Enter saves back through
``_queue_edit`` and never submits to the model, Ctrl+X deletes, Esc leaves the queue
untouched), the numbered picker that goes with it, and the double-Esc draft-discard
gesture. ``cli_tui_mixin`` stays the facade for the key-binding build and the Enter
handler, which delegate into this mixin.

Ctrl+E behaviour, in one place
------------------------------
*Press once, queue non-empty* — prints a numbered menu of every queued prompt and
loads the first one into the composer. While the menu is showing, a bare digit
``1``–``9`` selects that queued prompt (the ``Ctrl+E <n>`` the menu advertises —
see ``_tui_bind_queue_edit_keys`` for why it is sequenced rather than chorded).
*Press again* — cycles to the next queued prompt.
*Press with an empty queue* — restores the most recently submitted prompt from the
CLI's own history file so it can be edited and re-run. Enter then submits through the
normal execution path; nothing is written back into the queue.

Numbering is 1-based, oldest first, and always mirrors the live queue order, so the
number printed beside an item is the number that selects it at that moment.
"""
from __future__ import annotations

from pathlib import Path

from agent.i18n import t
from prompt_toolkit.filters import Condition

_PREVIEW_CHARS = 72


class CLITuiComposerEditMixin:
    """Composer editing gestures over the prompt buffer and the queued prompts."""

    # ── helpers ──────────────────────────────────────────────────────────────
    def _queue_edit_preview(self, item) -> str:
        """Single-line, width-bounded preview of a queued prompt."""
        text = str(item).replace("\r", " ").replace("\n", " ").strip()
        if len(text) <= _PREVIEW_CHARS:
            return text
        return text[: _PREVIEW_CHARS - 1] + "…"

    def _queue_edit_text(self, item) -> str:
        """The queued prompt's EXACT text.

        ``cli_session_dock._prompt_text`` collapses whitespace because it feeds one-line
        previews; loading a multi-line prompt with it would silently flatten the user's
        newlines into spaces. Editing therefore reads the raw value.
        """
        if isinstance(item, tuple):          # (text, images) from an image-attached submit
            item = item[0]
        text = getattr(item, "display_text", None) or getattr(item, "text", None) or str(item)
        return str(text)

    def _queue_edit_last_submitted(self) -> str:
        """Most recent prompt the user actually submitted, from the CLI history file.

        Slash commands are skipped: editing ``/model`` back into the composer is not
        what Ctrl+E is for. Returns "" when the history file is absent or empty.
        """
        path = getattr(self, "_history_file", None)
        if not path:
            return ""
        try:
            raw = Path(str(path)).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        # FileHistory layout: "\n# <timestamp>\n" then one "+<line>" per line of the entry.
        entries, current = [], []
        for line in raw.splitlines():
            if line.startswith("#"):
                if current:
                    entries.append("\n".join(current))
                    current = []
            elif line.startswith("+"):
                current.append(line[1:])
        if current:
            entries.append("\n".join(current))
        for entry in reversed(entries):          # newest first
            text = (entry or "").strip()
            if text and not text.startswith("/"):
                return entry.rstrip()
        return ""

    # ── gestures ─────────────────────────────────────────────────────────────
    def _tui_handle_double_escape(self, event):
        """Double ESC discards the draft and attached images (Claude Code / Gemini CLI gesture).

        Works while the agent streams — the gap Ctrl+C leaves (it interrupts the turn and only
        clears the draft when idle). The draft is appended to history first so Up recalls it,
        which is what makes a reflex key safe. Single ESC is the Alt-sequence prefix
        (escape+enter/g/v) so the escape-timeout keeps those distinct; modal prompts bind ESC
        eagerly and are excluded so cancel still wins.
        """
        buf = event.app.current_buffer
        if not (buf.text or self._attached_images):
            return
        buf.reset(append_to_history=bool(buf.text))
        self._attached_images.clear()
        event.app.invalidate()

    def _tui_handle_ctrl_e(self, event):
        """Ctrl+E: open the editing workflow.

        Queued prompts win when any exist — the numbered menu is printed and the first
        item is loaded. With an empty queue the last submitted prompt is restored
        instead, which is the case that used to look like a dead key: the old handler
        printed one line and returned without anything becoming editable.
        """
        from hermes_cli.cli_subagent_monitor import modal_prompt_active
        from cli import _cprint

        if modal_prompt_active(self) or getattr(self, "_connection_state", None):
            # Say why instead of returning in silence: the keystroke used to be filtered out at the
            # keyboard level, so a modal prompt made Ctrl+E look broken with no feedback at all.
            _cprint(f"  {t('cli.queue.edit_blocked_modal')}")
            return

        self._queue_edit_picker = False
        self._queue_edit_history = False

        items = self._pending_input_items()
        if items:
            current = getattr(self, "_queue_edit_index", None)
            if isinstance(current, int) and 1 <= current <= len(items):
                idx = current + 1 if current < len(items) else 1
            else:
                idx = 1
            self._queue_edit_render_menu(items, idx)
            self._queue_edit_load(event, idx, items)
            return

        if self._queue_edit_load_last_submitted(event):
            return

        _cprint(f"  {t('cli.queue.edit_nothing')}")

    def _queue_edit_render_menu(self, items, selected: int) -> None:
        """Print the numbered menu: every queued prompt with the shortcut that selects it."""
        from cli import _cprint
        _cprint(f"  {t('cli.queue.edit_menu_header', count=len(items))}")
        for number, item in enumerate(items, 1):
            marker = "▸" if number == selected else " "
            _cprint(f"   {marker} Ctrl+E {number}  {self._queue_edit_preview(item)}")

    def _queue_edit_load(self, event, idx: int, items, arm_picker: bool = True) -> None:
        """Put queued item ``idx`` into the composer.

        ``arm_picker`` is False when the load came *from* the picker: selecting a number
        closes the menu, it does not reopen it.
        """
        from cli import _cprint
        text = self._queue_edit_text(items[idx - 1])
        buf = event.app.current_buffer
        buf.text = text
        buf.cursor_position = len(text)
        self._queue_edit_index = idx
        self._queue_edit_picker = bool(arm_picker)   # digits 1-9 select while this is True
        self._queue_edit_history = False
        _cprint(f"  {t('cli.queue.edit_hint', index=idx, count=len(items))}")
        event.app.invalidate()

    def _queue_edit_load_last_submitted(self, event) -> bool:
        """Restore the last submitted prompt into the composer; False when there is none."""
        from cli import _cprint
        text = self._queue_edit_last_submitted()
        if not text:
            return False
        buf = event.app.current_buffer
        buf.text = text
        buf.cursor_position = len(text)
        self._queue_edit_index = None            # Enter must submit, not save back
        self._queue_edit_picker = False
        self._queue_edit_history = True
        _cprint(f"  {t('cli.queue.edit_history_restored')}")
        event.app.invalidate()
        return True

    def _queue_edit_digit_handler(self, number: int):
        """Bind ``number`` to selecting that queued prompt while the menu is showing."""

        def handler(event):
            from cli import _cprint
            items = self._pending_input_items()
            self._queue_edit_picker = False
            if not items:
                _cprint(f"  {t('cli.queue.edit_nothing')}")
                return
            if 1 <= number <= len(items):
                self._queue_edit_load(event, number, items, arm_picker=False)
            else:
                _cprint(f"  {t('cli.queue.edit_select_invalid', index=number, count=len(items))}")

        handler.__name__ = f"_tui_queue_edit_select_{number}"
        return handler

    def _tui_handle_queue_edit_enter(self, event) -> bool:
        """Enter dispatch while a queued prompt is being edited; True when consumed."""
        if getattr(self, "_queue_edit_index", None) is None:
            # Either no session, or a history-restored prompt: both submit normally.
            self._queue_edit_history = False
            return False
        self._tui_handle_queue_edit_save(event)
        return True

    def _tui_handle_queue_edit_save(self, event) -> bool:
        """Enter while editing a queued prompt: save the composer text back into the queue.

        Never submits to the model. Delegates to ``_queue_edit`` so item-not-found and the
        voice sentinel behave exactly like ``/queue edit N <prompt>`` (which also prints the
        confirmation); the composer and the edit flag are cleared either way.
        """
        from cli import _cprint
        idx = getattr(self, "_queue_edit_index", None)
        if idx is None:
            return False
        buf = event.app.current_buffer
        text = self._expand_paste_references(buf.text.strip())
        if not text:
            _cprint(f"  {t('cli.queue.edit_blank')}")
            return True
        self._queue_edit(f"{idx} {text}")
        buf.reset()
        self._queue_edit_index = None
        self._queue_edit_picker = False
        event.app.invalidate()
        return True

    def _tui_handle_queue_edit_cancel(self, event):
        """Esc while editing: clear the composer, leave the queue untouched."""
        from cli import _cprint
        active = (getattr(self, "_queue_edit_index", None) is not None
                  or getattr(self, "_queue_edit_picker", False)
                  or getattr(self, "_queue_edit_history", False))
        self._queue_edit_index = None
        self._queue_edit_picker = False
        self._queue_edit_history = False
        event.app.current_buffer.reset()
        if active:
            _cprint(f"  {t('cli.queue.edit_cancelled')}")
        event.app.invalidate()

    def _tui_handle_queue_edit_delete(self, event):
        """Ctrl+X while editing a queued prompt: remove it from the queue (like /queue rm N)."""
        idx = getattr(self, "_queue_edit_index", None)
        if idx is None:
            return
        self._queue_edit_index = None
        self._queue_edit_picker = False
        event.app.current_buffer.reset()
        self._queue_remove(str(idx))
        event.app.invalidate()

    def _tui_bind_queue_edit_keys(self, kb) -> None:
        """Bind Ctrl+E, the picker digits and the editing-session Escape/Ctrl+X overrides.

        Called last from ``_tui_build_key_bindings`` so the filtered bindings win over the
        generic Escape handlers (prompt_toolkit picks the last matching binding).

        Ctrl+E is bound with **no** filter. It used to carry ``not modal_prompt_active(self)``,
        and a filter that is False makes prompt_toolkit drop the keystroke without calling
        anything — the handler never ran, nothing was printed, and the key looked dead. The
        handler now runs and says why it cannot edit, so every press produces either the
        text in the composer or a visible reason.

        Why ``Ctrl+E`` then a digit instead of a ``Ctrl+E-1`` chord: a terminal delivers
        Ctrl+E as a single byte and a following digit as its own keypress, so prompt_toolkit
        can only express that as the two-key sequence ``kb.add("c-e", "1")`` — which makes
        every plain Ctrl+E press wait out the escape timeout for a second key that usually
        never comes. Digits are therefore armed only while the menu is showing
        (``_queue_edit_picker``), which is the same two keystrokes in the same order, with
        no latency on the common case and no collision with typing numbers into the query.
        """
        kb.add('c-e')(self._tui_handle_ctrl_e)
        _queue_edit_filter = Condition(
            lambda: getattr(self, '_queue_edit_index', None) is not None)
        _picker_filter = Condition(
            lambda: getattr(self, '_queue_edit_picker', False))
        _session_filter = Condition(
            lambda: (getattr(self, '_queue_edit_index', None) is not None
                     or getattr(self, '_queue_edit_picker', False)
                     or getattr(self, '_queue_edit_history', False)))
        kb.add('escape', filter=_session_filter, eager=True)(
            self._tui_handle_queue_edit_cancel)
        kb.add('c-x', filter=_queue_edit_filter)(self._tui_handle_queue_edit_delete)
        for number in range(1, 10):
            kb.add(str(number), filter=_picker_filter)(
                self._queue_edit_digit_handler(number))
