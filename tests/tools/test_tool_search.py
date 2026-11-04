"""Tests for tools/tool_search.py — progressive tool disclosure.

Coverage targets — these mirror the issues called out in the OpenClaw tool
search report. Every test that names an OpenClaw issue is the regression
guard that would have caught that specific failure mode.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:

    def test_defer_default_is_the_registered_list_and_a_user_list_replaces_it(self, caplog):
        """#116404: the curated deferral set lives in DEFAULT_CONFIG (so ``hermes config set``
        recognizes the key); a user list replaces it wholesale, [] keeps every tool eager, and a
        scalar is warned about (naming the expected shape) before falling back to the default."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from tools.tool_search import ToolSearchConfig, _DEFAULT_DEFERRED_TOOLS

        configured = frozenset(DEFAULT_CONFIG["tools"]["tool_search"]["defer"])
        assert isinstance(DEFAULT_CONFIG["tools"]["tool_search"]["defer"], list) and configured
        assert _DEFAULT_DEFERRED_TOOLS == configured
        assert ToolSearchConfig.from_raw(None).effective_defer_tools == configured
        assert ToolSearchConfig.from_raw({"defer": ["terminal"]}).effective_defer_tools == {"terminal"}
        assert ToolSearchConfig.from_raw({"defer": []}).effective_defer_tools == set()

        with caplog.at_level("WARNING", logger="tools.tool_search"):
            assert ToolSearchConfig.from_raw({"defer": "todo_list"}).effective_defer_tools == configured
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"


    def test_search_limits_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "search_default_limit": 999,
            "max_search_limit": 999,
        })
        assert cfg.max_search_limit == 50
        assert cfg.search_default_limit <= cfg.max_search_limit


# ---------------------------------------------------------------------------
# Classification — the hard invariant: core tools NEVER defer.
# ---------------------------------------------------------------------------


class TestClassification:
    def test_core_tools_never_defer(self):
        """The critical invariant from the OpenClaw report."""
        from tools.tool_search import is_deferrable_tool_name
        from toolsets import _HERMES_CORE_TOOLS
        assert _HERMES_CORE_TOOLS
        for core_name in _HERMES_CORE_TOOLS:
            assert not is_deferrable_tool_name(core_name), (
                f"Core tool '{core_name}' must NEVER be deferrable"
            )

    def test_bridge_tools_never_defer(self):
        from tools.tool_search import is_deferrable_tool_name, BRIDGE_TOOL_NAMES
        for name in BRIDGE_TOOL_NAMES:
            assert not is_deferrable_tool_name(name)

    def test_gui_surface_tools_never_defer(self):
        """Session-gated GUI tools stay direct and stay off the global core list."""
        from tools.registry import discover_builtin_tools, registry
        from tools.tool_search import _DIRECT_SURFACE_TOOLSETS, is_deferrable_tool_name
        from toolsets import _HERMES_CORE_TOOLS

        discover_builtin_tools()
        surface = [n for n, ts in registry.get_tool_to_toolset_map().items()
                   if ts in _DIRECT_SURFACE_TOOLSETS]
        assert surface
        for name in surface:
            assert not is_deferrable_tool_name(name), name
            assert name not in _HERMES_CORE_TOOLS


    def test_defer_override_restores_legacy_direct_gui(self):
        """tools.tool_search.defer: [] restores the everything-eager legacy:
        GUI tools alone no longer activate the bridge."""
        from tools.registry import discover_builtin_tools
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        discover_builtin_tools()
        names = {"read_window_below", "apply_layout", "project_list"}
        assembled = assemble_tool_defs(
            [_td(name, f"GUI {name}") for name in names],
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on", "defer": []}),
        )
        assert not assembled.activated
        assert {td["function"]["name"] for td in assembled.tool_defs} == names

    def test_core_working_set_never_defers_even_with_mcp_active(self):
        """The bridge activates for MCP, but working-set core tools (terminal,
        files, memory...) stay direct — the deferral set is the CURATED list,
        not all of core."""
        from tools.registry import discover_builtin_tools, registry
        from tools.tool_search import (
            BRIDGE_TOOL_NAMES,
            ToolSearchConfig,
            assemble_tool_defs,
        )

        discover_builtin_tools()
        mcp_name = "mcp_gui_surface_probe"
        registry.register(
            name=mcp_name,
            handler=lambda args, **kw: "{}",
            schema=_td(mcp_name, "Deferred MCP capability")["function"],
            toolset="mcp-gui-surface-probe",
        )

        assembled = assemble_tool_defs(
            [
                _td("terminal", "Run a command"),
                _td("memory", "Persistent memory"),
                _td("computer_use", "Drive the OS"),
                _td(mcp_name, "Deferred MCP capability"),
            ],
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {td["function"]["name"] for td in assembled.tool_defs}

        assert assembled.activated
        assert mcp_name not in names
        assert BRIDGE_TOOL_NAMES <= names
        assert {"terminal", "memory"} <= names
        # computer_use IS in the curated defer set → behind the bridge.
        assert "computer_use" not in names



    def test_classify_keeps_unknown_in_visible(self):
        """A tool we can't classify stays visible — never silently dropped.

        This is the OpenClaw #84141 regression guard (cron lost ``exec``
        because it wasn't in the catalog).
        """
        from tools.tool_search import classify_tools
        # Build a tool def for something we don't have a registry entry for.
        defs = [_td("xx_unknown_tool", "Unknown tool")]
        visible, deferrable = classify_tools(defs)
        names = {(td.get("function") or {}).get("name") for td in visible}
        assert "xx_unknown_tool" in names
        assert deferrable == []


# ---------------------------------------------------------------------------
# Token estimation + threshold gate
# ---------------------------------------------------------------------------


class TestThresholdGate:
    def test_off_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "off"})
        assert not should_activate(cfg, deferrable_tokens=1_000_000, context_length=200_000)


    def test_token_estimate_proportional_to_schema_size(self):
        from tools.tool_search import estimate_tokens_from_schemas
        small = [_td("a", "x")]
        big = [_td(f"name_{i}", f"description for tool {i} " * 20,
                   {"q": {"type": "string", "description": "search query " * 10}})
               for i in range(10)]
        small_t = estimate_tokens_from_schemas(small)
        big_t = estimate_tokens_from_schemas(big)
        assert big_t > small_t * 10


# ---------------------------------------------------------------------------
# Retrieval (BM25, rarest-token admission)
# ---------------------------------------------------------------------------


class TestRetrieval:
    def _fake_catalog(self):
        """Build a catalog directly without touching the registry."""
        from tools.tool_search import CatalogEntry
        from tools.tool_search_catalog import _tokenize, _entry_search_text
        defs = [
            _td("github_create_issue", "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}}),
            _td("github_search_repos", "Search GitHub for matching repositories",
                {"query": {"type": "string"}}),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}),
            _td("calendar_create_event", "Add an event to the user's calendar",
                {"title": {"type": "string"}, "start": {"type": "string"}}),
        ]
        catalog = []
        for d in defs:
            fn = d["function"]
            e = CatalogEntry(
                name=fn["name"], description=fn["description"],
                schema=d, source="mcp", source_name="mcp-test",
            )
            e._tokens = _tokenize(_entry_search_text(d))
            catalog.append(e)
        return catalog

    def test_search_finds_relevant_tool(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "create a github issue", limit=3)
        names = [h.name for h in hits]
        assert names[0] == "github_create_issue"


    def test_search_respects_limit(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "github", limit=1)
        assert len(hits) <= 1


class TestRelevanceFloor:
    """Coverage floor layered on the rarest-token gate.

    The gate stops a query whose intent word no tool carries. It does not stop a long
    hunt whose every word exists SOMEWHERE in the catalog while no single tool carries
    more than one of them; those must return nothing rather than a plausible-looking
    list the model rephrases against forever.
    """

    def _catalog(self):
        from tools.tool_search import build_catalog
        defs = [
            _td("github_rerun_failed_workflow_run_jobs",
                "Re-run failed jobs in a workflow run",
                {"run_id": {"type": "string"}}),
            _td("github_create_issue", "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}}),
            _td("github_list_issues", "List issues in a repository",
                {"repo": {"type": "string"}}),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}),
            # Every hunt word below is answerable by SOME document, none by one
            # document — the production catalog shape behind the 216-search trace.
            _td("gist_save_snippet", "Save a shell command snippet as a gist",
                {"content": {"type": "string"}}),
            _td("codeql_scan", "Scan code for vulnerabilities and execute analysis",
                {"repo": {"type": "string"}}),
        ]
        return build_catalog(defs)

    def test_incidental_single_term_match_is_filtered(self):
        # Every term answerable (each in exactly one document), so the rarest-token
        # gate admits the tool sharing that one word; the floor must not.
        from tools.tool_search import search_catalog
        hits = search_catalog(self._catalog(), "run shell command execute code", limit=5)
        assert hits == []

    def test_short_queries_are_untouched(self):
        # Below 4 answerable terms wording legitimately differs by a word.
        from tools.tool_search import search_catalog
        hits = search_catalog(self._catalog(), "list issues", limit=5)
        assert any(h.name == "github_list_issues" for h in hits)
        hits = search_catalog(self._catalog(), "send message", limit=5)
        assert any(h.name == "slack_send_message" for h in hits)

    def test_long_query_with_real_coverage_still_matches(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(
            self._catalog(), "create issue github repository title", limit=5)
        assert hits and hits[0].name == "github_create_issue"
        assert all(h.name != "github_rerun_failed_workflow_run_jobs" for h in hits)

    def test_exact_name_match_bypasses_coverage(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._catalog(), "github_create_issue", limit=5)
        assert hits and hits[0].name == "github_create_issue"


# ---------------------------------------------------------------------------
# Assembly — the full passthrough/activate decision.
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_no_deferrable_returns_unchanged(self):
        """Pure-core toolset: pass-through, no bridge tools added."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        defs = [_td("terminal", "Run shell"), _td("read_file", "Read a file")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        assert not result.activated
        assert {t["function"]["name"] for t in result.tool_defs} == {"terminal", "read_file"}

    def test_idempotent_when_bridge_already_present(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        defs = [_td("terminal", "Run shell"), _td("tool_search", "old")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "off"}),
        )
        names = [(t["function"]["name"]) for t in result.tool_defs]
        # The pre-existing tool_search was stripped (it would be re-injected if
        # activation happened; here it didn't).
        assert "tool_search" not in names


# ---------------------------------------------------------------------------
# Bridge dispatch
# ---------------------------------------------------------------------------


class TestBridgeDispatch:
    def test_tool_search_requires_queries(self):
        from tools.tool_search import dispatch_tool_search
        result = dispatch_tool_search({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_tool_search_rejects_empty_and_overcap_queries(self):
        import tools.tool_search as tool_search

        cfg = tool_search.ToolSearchConfig.from_raw({})
        assert "error" in json.loads(tool_search.dispatch_tool_search(
            {"queries": []}, current_tool_defs=[], config=cfg))
        assert "error" in json.loads(tool_search.dispatch_tool_search(
            {"queries": ["  ", ""]}, current_tool_defs=[], config=cfg))
        over = ["q"] * (tool_search._MAX_QUERIES_PER_CALL + 1)
        parsed = json.loads(tool_search.dispatch_tool_search(
            {"queries": over}, current_tool_defs=[], config=cfg))
        assert "error" in parsed

    def test_empty_search_keeps_connected_sources_discoverable(self):
        from tools.registry import registry
        from tools.tool_search import dispatch_tool_search

        name = "recovery_catalog_create_record"
        tool_def = _td(name, "Create a record in the connected catalog service.")
        registry.register(
            name=name,
            handler=lambda args, **kwargs: "{}",
            schema=tool_def,
            toolset="mcp-recovery-catalog",
        )

        result = json.loads(dispatch_tool_search(
            {"queries": ["unrelated vocabulary"]},
            current_tool_defs=[tool_def],
        ))

        [group] = result["results"]
        assert group["query"] == "unrelated vocabulary"
        assert group["matches"] == []
        assert result["tools"] == {}
        assert result["total_available"] == 1
        assert group["available_sources"] == [
            {"name": "recovery-catalog", "tool_count": 1},
        ]
        assert group["hint"]
        assert "available_sources" not in result


    def test_resolve_underlying_call_parses_object_args(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "unknown_xxx",
            "arguments": {"foo": "bar"},
        })
        # Will fail classification because unknown_xxx isn't deferrable.
        assert err is not None


    def test_resolve_underlying_call_rejects_recursion(self):
        """tool_call cannot invoke tool_call itself."""
        from tools.tool_search import resolve_underlying_call, TOOL_CALL_NAME
        name, args, err = resolve_underlying_call({
            "name": TOOL_CALL_NAME,
            "arguments": {},
        })
        assert err is not None

    @pytest.mark.parametrize("raw_args", ["", "  \n", None])
    def test_resolve_underlying_call_treats_blank_arguments_as_no_arguments(self, raw_args):
        """An OpenAI-compatible gateway emitting ``arguments: ""`` for a parameterless deferred tool
        must execute with {} instead of looping on a JSON parse error (#83937); malformed
        non-blank arguments still fail closed."""
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({"calls": [{"name": "todo_list", "arguments": raw_args}]})
        assert (name, args, err) == ("todo_list", {}, None)
        _, _, err = resolve_underlying_call({"calls": [{"name": "todo_list", "arguments": '{"todos": ['}]})
        assert err


# ---------------------------------------------------------------------------
# End-to-end via the real handle_function_call (smoke test).
# ---------------------------------------------------------------------------


class TestHandleFunctionCallIntegration:

    def test_tool_search_emits_one_terminal_hook(self, monkeypatch):
        """Inline bridge results still complete the tool lifecycle."""
        import model_tools
        from hermes_cli import lifecycle
        from tools import tool_search

        events = []
        monkeypatch.setattr(
            lifecycle,
            "has_hook",
            lambda name: name == "post_tool_call",
        )
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda name, **kwargs: events.append((name, kwargs)),
        )
        monkeypatch.setattr(
            tool_search,
            "dispatch_tool_search",
            lambda *args, **kwargs: json.dumps({"results": []}),
        )

        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"queries": ["private-query"]},
            session_id="private-session",
            task_id="private-task",
            turn_id="private-turn",
            api_request_id="private-request",
            tool_call_id="private-call",
        )

        assert json.loads(result) == {"results": []}
        assert len(events) == 1
        hook_name, payload = events[0]
        assert hook_name == "post_tool_call"
        assert payload["status"] == "ok"
        assert payload["turn_id"] == "private-turn"
        assert payload["api_request_id"] == "private-request"
        assert payload["tool_call_id"] == "private-call"


class TestRegression_OpenClawCron84141:
    """Regression guard for the OpenClaw cron-tool-loss class of bug.

    OpenClaw #84141: ``toolsAllow: ["exec"]`` on an isolated cron turn
    resulted in the agent receiving only ``sessions_send`` — the catalog
    builder silently dropped the requested core tool.

    Our defense: core tools are NEVER deferred. This test exercises the
    full assembly pipeline with a mixed core+MCP toolset and asserts that
    every core tool survives.
    """


    def test_unwrap_rejects_core_tool_attempt(self):
        """Even if the model tries to invoke a core tool through tool_call,
        we reject the call and tell the model to use it directly."""
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "terminal",
            "arguments": {"command": "echo hi"},
        })
        assert err is not None


class TestRegression_ToolsetScoping:
    """A restricted-toolset session must not see or invoke out-of-scope tools.

    The bug: the bridge dispatch and the tool_executor unwrap read the
    catalog from the *global* registry (get_tool_definitions with no
    toolset scope = "start with everything"), so a session scoped to one
    MCP server could tool_search the entire process registry and tool_call
    any plugin tool it was never granted. registry.dispatch() has no
    enabled_tools gate for non-execute_code tools, so the out-of-scope tool
    actually ran.

    The fix threads the session's enabled/disabled toolsets into the bridge
    dispatch (model_tools.handle_function_call) and the executor unwrap
    (agent.tool_executor), scoping both the searchable catalog and the
    invocable set to the session's own toolsets.
    """

    @staticmethod
    def _register(name, toolset):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True, "tool": name})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, f"desc for {name}", {"repo": {"type": "string"}}),
            toolset=toolset,
        )

    def test_search_catalog_is_scoped_to_session_toolsets(self):
        import model_tools

        for i in range(12):
            self._register(f"mcp_scoped_gh_{i}", "mcp-scoped-gh")
        self._register("scoped_oos_plugin", "scopedoosplugin")

        # tool_search scoped to the github toolset must not count the
        # out-of-scope plugin tool (or any of the host registry).
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"queries": ["mcp_scoped_gh"], "limit": 5},
            enabled_toolsets=["mcp-scoped-gh"],
        )
        parsed = json.loads(result)
        assert parsed["total_available"] == 12, (
            f"expected scoped catalog of 12, got {parsed['total_available']} "
            "— catalog leaked tools outside the session's toolsets"
        )
        hit_names = set(parsed["tools"])
        assert hit_names == {n for g in parsed["results"] for n in g["matches"]}
        assert "scoped_oos_plugin" not in hit_names


    def test_scoped_deferrable_names_helper(self):
        from tools.tool_search import scoped_deferrable_names

        self._register("mcp_helper_op", "mcp-helper")
        import model_tools
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-helper"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = scoped_deferrable_names(defs)
        assert "mcp_helper_op" in names
        # core tools are never deferrable
        assert "terminal" not in names


# ---------------------------------------------------------------------------
# Catalog listing (skills-style progressive disclosure)
# ---------------------------------------------------------------------------


class TestCatalogListing:


    def test_default_listing_cap_bounds_fixed_catalog_overhead(self):
        """The default manifest must not grow back to the old 20K-token cap."""
        from tools.registry import registry
        from tools.tool_search import (
            ToolSearchConfig,
            assemble_tool_defs,
            estimate_tokens_from_schemas,
        )

        defs = []
        for i in range(500):
            name = f"lean_catalog_tool_{i:04d}"
            registry.register(
                name=name,
                handler=lambda args, **kwargs: "{}",
                schema=_td(name, "Perform a deliberately verbose connected service action."),
                toolset="mcp-lean-catalog",
            )
            defs.append(_td(name, "Perform a deliberately verbose connected service action."))

        cfg = ToolSearchConfig.from_raw(None)
        result = assemble_tool_defs(defs, context_length=1_000_000, config=cfg)
        search = next(
            td for td in result.tool_defs
            if td["function"]["name"] == "tool_search"
        )
        description_tokens = estimate_tokens_from_schemas([search])
        # Includes the bridge schema around the listing, so allow modest
        # framing overhead above the 4K listing budget.
        assert description_tokens < 4500
        assert result.listing_form in {"names", "groups", "mixed"}



    @staticmethod
    def _register(name):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, "Deferred capability description.")["function"],
            toolset="mcp-listingtest",
        )


    def test_assembly_listing_off_keeps_legacy_description(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        for i in range(30):
            self._register(f"mcp_x_{i}")
        defs = [_td(f"mcp_x_{i}", "Deferred.") for i in range(30)]
        result = assemble_tool_defs(
            defs, context_length=1000,
            config=ToolSearchConfig.from_raw({"enabled": "on", "listing": "off"}),
        )
        assert result.activated
        search = next(t for t in result.tool_defs if t["function"]["name"] == "tool_search")
        assert "mcp_x_0" not in search["function"]["description"]


class TestDeferredCallSchemaProbe:
    """Blind tool_call invocations missing required arguments must return
    the tool's parameter schema instead of dispatching into an opaque
    downstream failure (port of nearai/ironclaw#5149's describe-first fix).

    A deferred tool's schema is invisible until tool_describe is called, so
    models routinely invoke deferred tools by name alone. Pre-fix, that
    produced ``KeyError: 'document_id'``-style errors that teach the model
    nothing; post-fix, the probe returns the schema so the model repairs
    the call in one round-trip. Valid calls dispatch untouched.
    """

    @staticmethod
    def _register(name, toolset, required=("document_id",)):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            # Simulates a tool that crashes opaquely on a missing required arg.
            return json.dumps({"ok": True, "doc": args["document_id"]})

        params = {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Doc id"},
                "format": {"type": "string"},
            },
            "required": list(required),
        }
        registry.register(
            name=name,
            handler=_handler,
            schema={"name": name, "description": f"desc {name}",
                    "parameters": params},
            toolset=toolset,
        )

    @staticmethod
    def _register_schema(name, toolset, params, calls):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            calls.append(args)
            return json.dumps({"ok": True, "args": args})

        registry.register(
            name=name,
            handler=_handler,
            schema={"name": name, "description": f"desc {name}",
                    "parameters": params},
            toolset=toolset,
        )

    def test_validator_returns_schema_for_missing_required(self):
        from tools.tool_search import validate_deferred_call_args

        self._register("mcp_probe_docs_get", "mcp-probe")
        err = validate_deferred_call_args("mcp_probe_docs_get", {})
        assert err is not None
        parsed = json.loads(err)
        assert "document_id" in parsed["error"]
        assert parsed["parameters"]["required"] == ["document_id"]
        assert "document_id" in parsed["parameters"]["properties"]


    def test_validator_never_blocks_unvalidatable_tools(self):
        from tools.tool_search import validate_deferred_call_args

        # Unknown tool → no schema → dispatch (downstream scope gate handles it).
        assert validate_deferred_call_args("mcp_no_such_tool_xyz", {}) is None


    def test_valid_tool_call_still_dispatches(self):
        import model_tools

        self._register("mcp_probe_valid_op", "mcp-probe-valid")
        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp_probe_valid_op",
                           "arguments": {"document_id": "abc"}},
            enabled_toolsets=["mcp-probe-valid"],
        ))
        assert result.get("ok") is True
        assert result.get("doc") == "abc"

    def test_invalid_enum_is_blocked_before_dispatch(self):
        import model_tools

        calls = []
        name = "mcp_probe_enum_validation"
        toolset = "mcp-probe-enum-validation"
        self._register_schema(name, toolset, {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["low", "high"]},
            },
            "required": ["priority"],
        }, calls)

        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": name, "arguments": {"priority": "urgent"}},
            enabled_toolsets=[toolset],
        ))

        assert calls == []
        assert result["path"] == "arguments.priority"
        assert result["constraint"] == "enum"

    @pytest.mark.parametrize(
        ("suffix", "arguments", "expected_path", "expected_constraint"),
        [
            (
                "nested_type",
                {"options": {"count": "not-an-integer"}},
                "arguments.options.count",
                "type",
            ),
            (
                "nested_required",
                {"options": {}},
                "arguments.options",
                "required",
            ),
            (
                "nested_extra",
                {"options": {"count": 1, "extra": True}},
                "arguments.options",
                "additionalProperties",
            ),
        ],
    )
    def test_validator_reports_nested_constraint_path(
        self, suffix, arguments, expected_path, expected_constraint,
    ):
        from tools.tool_search import validate_deferred_call_args

        calls = []
        name = f"mcp_probe_{suffix}"
        self._register_schema(name, "mcp-probe-nested", {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            },
            "required": ["options"],
        }, calls)

        result = json.loads(validate_deferred_call_args(name, arguments))

        assert result["path"] == expected_path
        assert result["constraint"] == expected_constraint

    def test_coercible_arguments_validate_then_dispatch_repaired(self):
        import model_tools

        calls = []
        name = "mcp_probe_coercion_validation"
        toolset = "mcp-probe-coercion-validation"
        self._register_schema(name, toolset, {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }, calls)

        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": name, "arguments": {"count": "42"}},
            enabled_toolsets=[toolset],
        ))

        assert result["ok"] is True
        assert calls == [{"count": 42}]

    def test_nullable_extension_remains_accepted(self):
        import model_tools

        calls = []
        name = "mcp_probe_nullable_validation"
        toolset = "mcp-probe-nullable-validation"
        self._register_schema(name, toolset, {
            "type": "object",
            "properties": {"value": {"type": "string", "nullable": True}},
            "required": ["value"],
        }, calls)

        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": name, "arguments": {"value": None}},
            enabled_toolsets=[toolset],
        ))

        assert result["ok"] is True
        assert calls == [{"value": None}]

    def test_schema_normalization_preserves_literal_enum_objects(self):
        from tools.tool_search import validate_deferred_call_args

        calls = []
        name = "mcp_probe_literal_enum_validation"
        enum_value = {"nullable": True, "$ref": "literal-not-a-schema"}
        self._register_schema(name, "mcp-probe-literal-enum", {
            "type": "object",
            "properties": {"value": {"enum": [enum_value]}},
            "required": ["value"],
        }, calls)

        assert validate_deferred_call_args(name, {"value": enum_value}) is None

    def test_malformed_schema_fails_open(self):
        import model_tools

        calls = []
        name = "mcp_probe_malformed_validation"
        toolset = "mcp-probe-malformed-validation"
        self._register_schema(name, toolset, {
            "type": "object",
            "properties": {"value": {"type": "not-a-json-schema-type"}},
        }, calls)

        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": name, "arguments": {"value": "kept"}},
            enabled_toolsets=[toolset],
        ))

        assert result["ok"] is True
        assert calls == [{"value": "kept"}]

    def test_external_ref_fails_open_without_resolution(self):
        from tools.tool_search import validate_deferred_call_args

        calls = []
        name = "mcp_probe_external_ref_validation"
        self._register_schema(name, "mcp-probe-external-ref", {
            "type": "object",
            "properties": {
                "payload": {"$ref": "https://example.invalid/schema.json"},
            },
        }, calls)

        assert validate_deferred_call_args(name, {"payload": {"anything": True}}) is None


# ---------------------------------------------------------------------------
# Recorded-outcome reliability weighting
# ---------------------------------------------------------------------------


def _reliability_catalog():
    """Two equally queryable github tools plus an unrelated one, built without the registry."""
    from tools.tool_search_catalog import build_catalog
    return build_catalog([
        _td("github_create_issue", "Open a new issue in a GitHub repository",
            {"title": {"type": "string"}, "body": {"type": "string"}}),
        _td("github_search_repos", "Search GitHub for matching repositories",
            {"query": {"type": "string"}}),
        _td("slack_send_message", "Post a message into a Slack channel",
            {"channel": {"type": "string"}, "text": {"type": "string"}}),
    ])


def _write_messages_db(path, rows, now: float = 1_700_000_000.0):
    """A minimal ``messages`` table (the real one's shape for these columns) plus *rows*.

    ``rows`` are ``(tool_name, content, age_seconds)``, inserted oldest first so the rowid order
    the loader reads newest-first matches their timestamps.
    """
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,"
            " role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp)"
            " VALUES ('s1', 'tool', ?, ?, ?)",
            [(content, name, now - age) for name, content, age in rows])
        conn.commit()
    finally:
        conn.close()
    return now


_OK = '{"output": "done", "exit_code": 0, "error": null}'
_EXIT_1 = '{"output": "", "exit_code": 1, "error": null}'


class TestReliabilityWeighting:
    """Recorded outcomes may re-order the hits lexical search admitted, never hide one.

    The multiplier is the narrow form of AnyTool's ``ToolQualityManager.adjust_ranking``: it is
    applied after the admission gate, clamped to ``[0.2, 1.0]``, and neutral without evidence.
    """

    def test_no_history_ranks_exactly_as_before(self):
        from tools.tool_search_catalog import search_catalog

        catalog = _reliability_catalog()
        for query in ("create a github issue", "github repositories", "post a slack message"):
            baseline = [e.name for e in search_catalog(catalog, query, limit=5)]
            assert baseline, f"{query!r} must admit something for this comparison to mean anything"
            assert [e.name for e in search_catalog(catalog, query, limit=5, reliability=None)] == baseline
            assert [e.name for e in search_catalog(catalog, query, limit=5, reliability={})] == baseline
            neutral = {e.name: 1.0 for e in catalog}
            assert [e.name for e in search_catalog(catalog, query, limit=5, reliability=neutral)] == baseline

    def test_history_below_the_evidence_threshold_changes_nothing(self, tmp_path):
        from tools.tool_search_catalog import (
            MIN_CALLS_FOR_RELIABILITY, load_tool_reliability, search_catalog)

        db = tmp_path / "state.db"
        now = _write_messages_db(db, [("github_create_issue", _EXIT_1, 30),
                                      ("github_create_issue", _EXIT_1, 20)])
        reliability = load_tool_reliability(db, now=now)
        assert reliability == {"github_create_issue": 1.0}  # 2 calls < MIN_CALLS_FOR_RELIABILITY

        catalog = _reliability_catalog()
        query = "github repositories"
        baseline = [e.name for e in search_catalog(catalog, query, limit=5)]
        assert [e.name for e in search_catalog(catalog, query, limit=5, reliability=reliability)] == baseline
        assert MIN_CALLS_FOR_RELIABILITY > 2

    def test_poor_history_demotes_a_tool_but_never_drops_it(self):
        from tools.tool_search_catalog import search_catalog

        catalog = _reliability_catalog()
        query = "github repositories"
        baseline = [e.name for e in search_catalog(catalog, query, limit=10)]
        assert baseline == ["github_search_repos", "github_create_issue"]

        weighted = [e.name for e in search_catalog(
            catalog, query, limit=10, reliability={"github_search_repos": 0.2})]
        assert sorted(weighted) == sorted(baseline)  # every admitted tool still offered
        assert weighted == ["github_create_issue", "github_search_repos"]  # just re-ordered

    def test_a_zero_multiplier_from_the_caller_cannot_hide_a_tool(self):
        from tools.tool_search_catalog import search_catalog

        catalog = _reliability_catalog()
        query = "github repositories"
        baseline = [e.name for e in search_catalog(catalog, query, limit=10)]
        for bad in (0.0, -3.0, float("-inf"), "junk", None):
            names = [e.name for e in search_catalog(
                catalog, query, limit=10, reliability={"github_search_repos": bad})]
            assert sorted(names) == sorted(baseline), f"{bad!r} must not remove a hit"

    def test_exact_name_hit_keeps_first_place(self):
        from tools.tool_search_catalog import search_catalog

        catalog = _reliability_catalog()
        hits = search_catalog(catalog, "github_create_issue", limit=5,
                              reliability={"github_create_issue": 0.2})
        assert hits[0].name == "github_create_issue"

    @pytest.mark.parametrize("successes,calls,streak", [
        (0, 3, 0), (0, 50, 50), (1, 100, 100), (2, 40, 3), (1, 3, 2), (3, 10, 5),
    ])
    def test_multiplier_stays_inside_the_clamp(self, successes, calls, streak):
        from tools.tool_search_catalog import RELIABILITY_FLOOR, tool_reliability_multiplier

        assert RELIABILITY_FLOOR <= tool_reliability_multiplier(successes, calls, streak) <= 1.0

    def test_a_tool_that_never_worked_bottoms_out_at_the_floor(self):
        from tools.tool_search_catalog import RELIABILITY_FLOOR, tool_reliability_multiplier

        assert tool_reliability_multiplier(0, 100, consecutive_failures=50) == RELIABILITY_FLOOR
        # 3 straight failures are penalised, but never past the floor the clamp enforces.
        assert tool_reliability_multiplier(0, 3, 3) >= RELIABILITY_FLOOR

    def test_a_healthy_tool_is_never_penalised(self):
        from tools.tool_search_catalog import tool_reliability_multiplier

        assert tool_reliability_multiplier(2, 5) == 1.0
        assert tool_reliability_multiplier(3, 3, 0) == 1.0


class TestLoadToolReliability:
    """The loader reads recorded outcomes once, from a bounded rowid window, and never raises."""

    def test_failures_lower_the_multiplier(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        now = _write_messages_db(db, [
            ("flaky_tool", _EXIT_1, 50), ("flaky_tool", _EXIT_1, 40),
            ("flaky_tool", _EXIT_1, 30), ("flaky_tool", _EXIT_1, 20),
            ("flaky_tool", _OK, 10),  # newest call worked: no consecutive-failure penalty
        ])
        reliability = load_tool_reliability(db, now=now)
        assert reliability["flaky_tool"] == pytest.approx(0.65)  # 1/5 success rate

    def test_a_consecutive_failure_run_penalises_further(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        rows = [("t", _OK, 50), ("t", _EXIT_1, 40), ("t", _EXIT_1, 30),
                ("t", _EXIT_1, 20), ("t", _EXIT_1, 10)]
        now = _write_messages_db(db, rows)
        reliability = load_tool_reliability(db, now=now)
        assert reliability["t"] < 0.65  # same 1/5 rate, but the last 4 calls all failed

    def test_tools_without_history_are_absent(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        now = _write_messages_db(db, [("known_tool", _OK, 10)])
        assert load_tool_reliability(db, now=now) == {"known_tool": 1.0}

    def test_outcomes_outside_the_window_are_ignored(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        old = 60 * 86400
        now = _write_messages_db(db, [("stale_tool", _EXIT_1, old + 30),
                                      ("stale_tool", _EXIT_1, old + 20),
                                      ("stale_tool", _EXIT_1, old + 10)])
        assert load_tool_reliability(db, now=now) == {}

    def test_missing_database_is_empty_history(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        assert load_tool_reliability(tmp_path / "absent.db") == {}

    def test_database_without_a_messages_table_is_empty_history(self, tmp_path):
        import sqlite3

        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE sessions (id TEXT)")
            conn.commit()
        finally:
            conn.close()
        assert load_tool_reliability(db) == {}

    def test_exit_code_envelope_decides_success(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        now = _write_messages_db(db, [
            ("broken_tool", _EXIT_1, 30), ("broken_tool", _EXIT_1, 20),
            ("broken_tool", _EXIT_1, 10),
            ("working_tool", _OK, 30), ("working_tool", _OK, 20), ("working_tool", _OK, 10),
        ])
        reliability = load_tool_reliability(db, now=now)
        assert reliability["working_tool"] == 1.0
        assert reliability["broken_tool"] < 0.5

    def test_prose_error_results_count_as_failures(self, tmp_path):
        from tools.tool_search_catalog import load_tool_reliability

        db = tmp_path / "state.db"
        now = _write_messages_db(db, [("mcp_tool", "Error: connection refused", 30),
                                      ("mcp_tool", "Error: connection refused", 20),
                                      ("mcp_tool", "Error: connection refused", 10)])
        assert load_tool_reliability(db, now=now)["mcp_tool"] < 1.0


import tools.tool_search as tool_search  # noqa: E402


class TestZeroMatchLlmPass:
    """Opt-in recovery when lexical search returns nothing (AnyTool gap analysis, item 2).

    The pass must be inert unless asked for, must never surface a name the corpus does not
    contain, and must never turn a failed recovery into a failed search.
    """

    # Real names from the curated defer set: dispatch builds the catalog from the on-disk
    # config, so invented names would produce an empty catalog and prove nothing.
    DEFS = ("cronjob_manage", "image_generate")

    def _defs(self):
        return [
            _td("cronjob_manage", "Schedule and manage recurring background jobs"),
            _td("image_generate", "Generate an image from a text prompt"),
        ]

    def _cfg(self, **kw):
        return tool_search.ToolSearchConfig(
            enabled="on", threshold_pct=10, search_default_limit=5, max_search_limit=20, **kw)

    def test_inert_by_default(self):
        calls = []
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(),
            llm_select=lambda q, l: calls.append(q) or ["image_generate"]))
        assert calls == []
        assert out["results"][0]["matches"] == []

    def test_recovers_on_zero_match_when_enabled(self):
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True),
            llm_select=lambda q, l: ["image_generate"]))
        group = out["results"][0]
        assert group["matches"] == ["image_generate"]
        assert group["match_mode"] == "llm_name_pass"

    def test_hallucinated_name_is_filtered(self):
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True),
            llm_select=lambda q, l: ["totally_invented_tool"]))
        assert out["results"][0]["matches"] == []

    def test_selector_failure_does_not_break_search(self):
        def boom(q, l):
            raise RuntimeError("model down")
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True), llm_select=boom))
        assert out["results"][0]["matches"] == []

    def test_not_used_when_lexical_search_succeeds(self):
        calls = []
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["schedule a recurring job"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True),
            llm_select=lambda q, l: calls.append(q) or []))
        assert calls == []
        assert out["results"][0]["matches"]

    def test_flag_parsed_from_raw_config(self):
        cfg = tool_search.ToolSearchConfig.from_raw({"zero_match_llm": True})
        assert cfg.zero_match_llm is True
        assert tool_search.ToolSearchConfig.from_raw({}).zero_match_llm is False


class TestZeroMatchSelectorWiring:
    """Integration: the real selector, backed by the auxiliary client, actually gets invoked."""

    DEFS = ("cronjob_manage", "image_generate")

    def _defs(self):
        return [
            _td("cronjob_manage", "Schedule and manage recurring background jobs"),
            _td("image_generate", "Generate an image from a text prompt"),
        ]

    def _cfg(self, **kw):
        return tool_search.ToolSearchConfig(
            enabled="on", threshold_pct=10, search_default_limit=5, max_search_limit=20, **kw)

    def test_selector_calls_aux_client_and_recovers(self, monkeypatch):
        calls = []

        def fake_call_llm(**kw):
            calls.append(kw)
            return {"choices": [{"message": {"content": '["image_generate"]'}}]}

        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "call_llm", fake_call_llm, raising=True)

        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True),
            llm_select=tool_search.build_llm_selector()))
        assert out["results"][0]["matches"] == ["image_generate"]
        assert len(calls) == 1
        assert calls[0]["temperature"] == 0.0
        assert calls[0]["max_tokens"] == 200          # bounded
        assert "zzzqqq unmatched" in calls[0]["messages"][0]["content"]

    def test_lexical_success_makes_no_model_call(self, monkeypatch):
        calls = []
        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "call_llm",
                            lambda **kw: calls.append(kw) or {"choices": []}, raising=True)
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["schedule a recurring job"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True),
            llm_select=tool_search.build_llm_selector()))
        assert out["results"][0]["matches"]
        assert calls == []                            # no added cost on the normal path

    def test_flag_off_makes_no_model_call(self, monkeypatch):
        calls = []
        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "call_llm",
                            lambda **kw: calls.append(kw) or {"choices": []}, raising=True)
        tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(), llm_select=tool_search.build_llm_selector())
        assert calls == []

    def test_aux_failure_is_isolated(self, monkeypatch):
        import agent.auxiliary_client as aux

        def boom(**kw):
            raise RuntimeError("provider down")
        monkeypatch.setattr(aux, "call_llm", boom, raising=True)
        out = json.loads(tool_search.dispatch_tool_search(
            {"queries": ["zzzqqq unmatched"]}, current_tool_defs=self._defs(),
            config=self._cfg(zero_match_llm=True),
            llm_select=tool_search.build_llm_selector()))
        assert out["results"][0]["matches"] == []

    def test_object_shaped_response_parsed(self, monkeypatch):
        import agent.auxiliary_client as aux

        class _Msg:
            content = '["image_generate"]'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        monkeypatch.setattr(aux, "call_llm", lambda **kw: _Resp(), raising=True)
        assert tool_search.build_llm_selector()("q", "listing") == ["image_generate"]

    def test_non_json_reply_falls_back_to_identifiers(self, monkeypatch):
        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "call_llm",
                            lambda **kw: "I suggest image_generate please", raising=True)
        assert "image_generate" in tool_search.build_llm_selector()("q", "listing")
