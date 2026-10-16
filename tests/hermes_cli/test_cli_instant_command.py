"""`/instant` mid-execution redirect: argument parsing, busy dispatch, fallbacks.

/instant maps onto ``agent.redirect()`` while a turn is live (cancels only the
in-flight model request; during tool execution it steers after the current
tool). With no live turn — idle, no agent, no redirect support, or a rejected
redirect — it falls back to ``_pending_input`` as the next turn.
"""

from __future__ import annotations

from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prompt_toolkit.buffer import Buffer

from cli import HermesCLI
from hermes_cli.cli_loops_mixin import _parse_instant_payload


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._agent_running = False
    cli.agent = None
    cli._pending_input = Queue()
    cli._pending_resume_sessions = None
    return cli


def _queued(cli) -> list:
    with cli._pending_input.mutex:
        return list(cli._pending_input.queue)


class TestInstantArgParsing:

    def test_double_quoted(self):
        assert _parse_instant_payload('"fix the bug"') == "fix the bug"

    def test_single_quoted(self):
        assert _parse_instant_payload("'fix the bug'") == "fix the bug"

    def test_unquoted(self):
        assert _parse_instant_payload("fix the bug") == "fix the bug"

    def test_unbalanced_quotes_fall_through_as_literal_text(self):
        assert _parse_instant_payload('"fix the bug') == '"fix the bug'
        assert _parse_instant_payload('fix the bug"') == 'fix the bug"'
        assert _parse_instant_payload("\"mismatched'") == "\"mismatched'"

    def test_internal_whitespace_and_newlines_survive(self):
        assert _parse_instant_payload('"line one\nline two"') == "line one\nline two"
        assert _parse_instant_payload("  spaced  out  ") == "spaced  out"

    def test_empty_forms(self):
        for raw in ("", "   ", '""', "''"):
            assert _parse_instant_payload(raw) == ""

    def test_only_one_pair_of_quotes_is_stripped(self):
        assert _parse_instant_payload('""hello""') == '"hello"'


class TestInstantBusyRedirect:

    def test_accepted_redirect_calls_agent_with_exact_payload_and_does_not_queue(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli.agent.redirect = MagicMock(return_value=True)
        with patch("cli._cprint"):
            assert cli.process_command('/instant "focus on error handling"') is True
        cli.agent.redirect.assert_called_once_with("focus on error handling")
        assert _queued(cli) == []

    def test_rejected_redirect_falls_back_to_the_queue(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli.agent.redirect = MagicMock(return_value=False)
        with patch("cli._cprint"):
            cli.process_command("/instant focus")
        cli.agent.redirect.assert_called_once_with("focus")
        assert _queued(cli) == ["focus"]

    def test_redirect_exception_falls_back_to_the_queue(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli.agent.redirect = MagicMock(side_effect=RuntimeError("boom"))
        with patch("cli._cprint") as cprint:
            cli.process_command("/instant focus")
        assert _queued(cli) == ["focus"]
        assert any("Redirect failed" in str(call) for call in cprint.call_args_list)

    def test_idle_queues_without_touching_the_agent(self):
        cli = _cli()
        cli._agent_running = False
        cli.agent = MagicMock()
        with patch("cli._cprint"):
            cli.process_command('/instant "next turn"')
        cli.agent.redirect.assert_not_called()
        assert _queued(cli) == ["next turn"]

    def test_agent_without_redirect_support_falls_back_to_the_queue(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = SimpleNamespace()
        with patch("cli._cprint"):
            cli.process_command("/instant do it")
        assert _queued(cli) == ["do it"]

    def test_no_agent_falls_back_to_the_queue(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = None
        with patch("cli._cprint"):
            cli.process_command("/instant do it")
        assert _queued(cli) == ["do it"]

    def test_empty_payload_prints_usage_and_queues_nothing(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        with patch("cli._cprint") as cprint:
            cli.process_command("/instant")
            cli.process_command('/instant ""')
            cli.process_command("/instant    ")
        cli.agent.redirect.assert_not_called()
        assert _queued(cli) == []
        assert sum("Usage: /instant" in str(call) for call in cprint.call_args_list) == 3

    def test_repeated_accepted_invocations_do_not_duplicate(self):
        cli = _cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli.agent.redirect = MagicMock(return_value=True)
        with patch("cli._cprint"):
            cli.process_command("/instant one")
            cli.process_command("/instant two")
        assert cli.agent.redirect.call_count == 2
        assert _queued(cli) == []


class TestInstantInlineDetector:

    def test_detects_instant_while_the_agent_runs(self):
        cli = _cli()
        cli._agent_running = True
        assert cli._should_handle_instant_command_inline("/instant focus on tests") is True

    def test_ignores_the_word_instant_in_a_normal_message(self):
        cli = _cli()
        cli._agent_running = True
        assert cli._should_handle_instant_command_inline("instant focus") is False
        assert cli._should_handle_instant_command_inline("the instant it arrives, stop") is False

    def test_ignores_when_the_agent_is_idle(self):
        cli = _cli()
        cli._agent_running = False
        assert cli._should_handle_instant_command_inline("/instant focus") is False

    def test_ignores_other_slash_commands(self):
        cli = _cli()
        cli._agent_running = True
        assert cli._should_handle_instant_command_inline("/steer hello") is False
        assert cli._should_handle_instant_command_inline("/queue hello") is False
        assert cli._should_handle_instant_command_inline("/bg hello") is False

    def test_ignores_attached_images(self):
        cli = _cli()
        cli._agent_running = True
        assert cli._should_handle_instant_command_inline("/instant look", has_images=True) is False


class TestInstantDispatchWiring:

    def test_enter_routes_instant_inline_while_busy(self):
        cli = _cli()
        cli._agent_running = True
        cli.process_command = MagicMock(return_value=True)
        cli._should_handle_model_command_inline = lambda text, has_images=False: False
        cli._should_handle_steer_command_inline = lambda text, has_images=False: False
        cli._should_handle_background_command_inline = lambda text, has_images=False: False
        buf = Buffer()
        event = SimpleNamespace(
            app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=False))
        assert cli._tui_enter_inline_command(event, "/instant go", False) is True
        cli.process_command.assert_called_once_with("/instant go")

    def test_registry_entry_is_cli_only_and_dispatches_while_busy(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("instant")
        assert cmd is not None
        assert cmd.name == "instant"
        assert cmd.cli_only is True
        assert cmd.busy_policy == "dispatch"
        assert cmd.args_hint == '"<query>"'

    def test_slash_dispatch_table_entry(self):
        assert HermesCLI._slash_handler("instant") == ("_cmd_instant", True)
