"""Composer editing gestures for the interactive CLI input area.

Topical sibling of ``cli_tui_mixin``: this mixin owns the Ctrl+E queued-prompt editing
session (``_queue_edit_index`` — load from ``_pending_input``, Enter saves back through
``_queue_edit`` and never submits to the model, Ctrl+X deletes, Esc leaves the queue
untouched) and the double-Esc draft-discard gesture. ``cli_tui_mixin`` stays the facade for
the key-binding build and the Enter handler, which delegate into this mixin.
"""
from __future__ import annotations

from agent.i18n import t
from prompt_toolkit.filters import Condition


class CLITuiComposerEditMixin:
    """Composer editing gestures over the prompt buffer and the queued prompts."""

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
        """Ctrl+E: load the next queued prompt into the composer for editing.

        While ``_queue_edit_index`` is set Enter SAVES back into the queue (never submits to
        the model), Ctrl+E cycles to the next queued prompt, Ctrl+X removes the one being
        edited, and Esc leaves the queue untouched.
        """
        from hermes_cli.cli_session_dock import _prompt_text
        from hermes_cli.cli_subagent_monitor import modal_prompt_active
        if modal_prompt_active(self) or getattr(self, "_connection_state", None):
            return
        from cli import _cprint
        items = self._pending_input_items()
        if not items:
            _cprint(f"  {t('cli.queue.edit_empty')}")
            return
        current = getattr(self, "_queue_edit_index", None)
        if isinstance(current, int) and 1 <= current <= len(items):
            idx = current + 1 if current < len(items) else 1
        else:
            idx = 1
        text = _prompt_text(items[idx - 1])
        buf = event.app.current_buffer
        buf.text = text
        buf.cursor_position = len(text)
        self._queue_edit_index = idx
        _cprint(f"  {t('cli.queue.edit_hint', index=idx, count=len(items))}")
        event.app.invalidate()

    def _tui_handle_queue_edit_enter(self, event) -> bool:
        """Enter dispatch while a queued prompt is being edited; True when consumed."""
        if getattr(self, "_queue_edit_index", None) is None:
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
        event.app.invalidate()
        return True

    def _tui_handle_queue_edit_cancel(self, event):
        """Esc while editing a queued prompt: clear the composer, leave the queue untouched."""
        from cli import _cprint
        self._queue_edit_index = None
        event.app.current_buffer.reset()
        _cprint(f"  {t('cli.queue.edit_cancelled')}")
        event.app.invalidate()

    def _tui_handle_queue_edit_delete(self, event):
        """Ctrl+X while editing a queued prompt: remove it from the queue (like /queue rm N)."""
        idx = getattr(self, "_queue_edit_index", None)
        if idx is None:
            return
        self._queue_edit_index = None
        event.app.current_buffer.reset()
        self._queue_remove(str(idx))
        event.app.invalidate()

    def _tui_bind_queue_edit_keys(self, kb) -> None:
        """Bind Ctrl+E and the editing-session Escape/Ctrl+X overrides.

        Called last from ``_tui_build_key_bindings`` so the filtered bindings win over the
        generic Escape handlers (prompt_toolkit picks the last matching binding).
        """
        from hermes_cli.cli_subagent_monitor import modal_prompt_active
        kb.add('c-e', filter=Condition(lambda: not modal_prompt_active(self)))(
            self._tui_handle_ctrl_e)
        _queue_edit_filter = Condition(
            lambda: getattr(self, '_queue_edit_index', None) is not None)
        kb.add('escape', filter=_queue_edit_filter, eager=True)(
            self._tui_handle_queue_edit_cancel)
        kb.add('c-x', filter=_queue_edit_filter)(self._tui_handle_queue_edit_delete)
