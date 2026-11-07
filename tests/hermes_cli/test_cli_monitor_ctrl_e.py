"""Ctrl+E from the subagent monitor: the queue bar advertises it, so it has to work there.

The monitor is a full-screen prompt_toolkit ``Application`` with its own key bindings. While it is
open, the REPL's bindings do not receive keystrokes at all — so the queue heading's "Ctrl+E edit"
went nowhere. The monitor now exits with ``result='queue-edit'`` and the caller replays the key
against ``cli._app``.
"""
from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.application import Application
from prompt_toolkit.output import DummyOutput

from hermes_cli.cli_subagent_monitor import _request_queue_edit, build_monitor_application


class _MonitorStub:
    def __init__(self):
        self.cli = SimpleNamespace()
        self.roster = []
        self.processes = []
        self.queued = []
        self.collapsed = False
        self.opening = False
        self.selected = 0
        self.app = None

    def __getattr__(self, name):
        # The monitor view reads optional state (title, detail rows, notices) that a stub need not
        # model; anything unset reads as empty rather than crashing the layout.
        if name.startswith("__"):
            raise AttributeError(name)
        return ""

    def refresh(self):
        pass

    def dock_text(self, columns=80, rows=24):
        return ""


def test_the_monitor_registers_ctrl_e_for_the_queue_edit():
    """The queue heading says "Ctrl+E edit"; the full-screen monitor must bind the key.

    A driving test of the whole monitor application needs a live roster, so this pins the one thing
    that was actually missing: the binding, pointing at the handler that exits with the edit result.
    """
    app = build_monitor_application(_MonitorStub())
    handlers = [b.handler.__name__ for b in app.key_bindings.bindings
                if getattr(b, "keys", None) == ("c-e",)]
    assert handlers == ["edit_queued"]


def test_request_queue_edit_replays_the_key_against_the_repl_app():
    calls = []

    class Shell:
        _app = Application(output=DummyOutput())
        def _tui_handle_ctrl_e(self, event):
            calls.append(event.app is self._app)

    _request_queue_edit(Shell())
    assert calls == [True], "the composer editor must run against the REPL application"


def test_request_queue_edit_is_a_no_op_without_a_running_app():
    _request_queue_edit(SimpleNamespace())   # must not raise
