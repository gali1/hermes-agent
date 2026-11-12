"""Provider contract tests for the Rekal memory plugin.

Ported from OpenCode's ``mempalace.test.ts`` and extended for the scoped tool
surface (E1/D13) and the store-backed diary (C1).

Three invariants are enforced mechanically here, because all three were real
defects in the previous single 32-operation mega-tool:

* every action a schema advertises has a handler, and no handler exists that no
  schema advertises (no drift in either direction),
* every parameter a handler reads is declared in its tool schema, so a value
  the model sends is never silently dropped, and
* no destructive or administrative operation is reachable through
  ``handle_tool_call()``.
"""

import inspect
import json
import os
import tempfile

# Redirect the MemPalace/ChromaDB side to a throwaway directory BEFORE any
# mempalace import (the provider imports it lazily at initialize()).
_PALACE = tempfile.mkdtemp(prefix="rekal-provider-test-palace-")
os.environ.setdefault("MEMPALACE_PALACE_PATH", _PALACE)
os.environ.setdefault("MEMPALACE_DATA_DIR", _PALACE)

import pytest  # noqa: E402

import plugins.memory.rekal as rekal_plugin  # noqa: E402
from plugins.memory.rekal import (  # noqa: E402
    LLM_ACTIONS,
    TOOL_SCHEMAS,
    _INTERNAL_OPERATIONS,
    _TOOL_SPECS,
    RekalMemoryProvider,
)

# The approved LLM-facing surface (item E1).  Adding to this set is a design
# decision, not a refactor: every entry must be additive or read-only.
APPROVED_SURFACE = {
    ("rekal_search", "search"),
    ("rekal_search", "recall"),
    ("rekal_search", "get"),
    ("rekal_remember", "store"),
    ("rekal_remember", "reinforce"),
    ("rekal_link", None),
    ("rekal_overview", "timeline"),
    ("rekal_overview", "topics"),
    ("rekal_overview", "contradictions"),
    ("rekal_overview", "stats"),
    ("rekal_ingest", None),
    ("rekal_diary", "write"),
    ("rekal_diary", "read"),
}

# Destructive / administrative operations: internal-only, never model-callable.
EXCLUDED_OPERATIONS = {
    "memory_delete",
    "memory_update",
    "memory_supersede",
    "deduplicate",
    "consolidate",
    "purge",
    "rebuild_fts",
    "migrate",
    "set_config",
    "session_init",
    "build_context",
    "memory_health",
}

_DESTRUCTIVE_VERBS = ("delete", "remove", "purge", "drop", "clear", "destroy",
                      "reset", "rebuild", "migrate", "config", "update", "supersede")


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def provider(tmp_path, monkeypatch):
    # Keep the heavy optional import out of the test path; the point of C1 is
    # that nothing on the engine/diary surface needs it.
    monkeypatch.setattr(rekal_plugin, "_load_mempalace", lambda: None)
    monkeypatch.setattr(rekal_plugin, "_mcp_mod", None)
    monkeypatch.setattr(rekal_plugin, "_MEMPALACE_IMPORT_ERROR", "not installed")
    p = RekalMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    yield p
    p.shutdown()


def call(provider, tool, action=None, **args):
    payload = dict(args)
    if action is not None:
        payload["action"] = action
    return provider.handle_tool_call(tool, payload)


def ok(provider, tool, action=None, **args):
    raw = call(provider, tool, action, **args)
    payload = json.loads(raw)
    assert "error" not in payload, payload
    return payload


def schema_for(tool_name):
    for schema in TOOL_SCHEMAS:
        if schema["name"] == tool_name:
            return schema
    raise AssertionError(f"no schema for {tool_name}")


# ── Schema contract ──────────────────────────────────────────────────────

def test_tool_schemas_are_well_formed():
    names = [schema["name"] for schema in TOOL_SCHEMAS]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"
    for schema in TOOL_SCHEMAS:
        assert schema["name"].startswith("rekal_"), schema["name"]
        assert schema["description"].strip()
        params = schema["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict) and params["properties"]
        for required in params["required"]:
            assert required in params["properties"], (schema["name"], required)


def test_tool_names_do_not_shadow_hermes_core_tools():
    from toolsets import _HERMES_CORE_TOOLS

    clashes = {schema["name"] for schema in TOOL_SCHEMAS} & set(_HERMES_CORE_TOOLS)
    assert not clashes, f"core tool names are reserved: {clashes}"


def test_enum_matches_dispatch_table_and_handlers_exist(provider):
    for tool_name, spec in _TOOL_SPECS.items():
        schema = schema_for(tool_name)
        actions = spec.get("actions")
        if actions:
            enum = schema["parameters"]["properties"]["action"]["enum"]
            assert list(enum) == list(actions), f"{tool_name}: enum/dispatch drift"
            assert "action" in schema["parameters"]["required"]
            for action, handler_name in actions.items():
                handler = getattr(provider, handler_name, None)
                assert callable(handler), f"{tool_name}/{action} -> missing {handler_name}"
        else:
            assert "action" not in schema["parameters"]["properties"]
            assert callable(getattr(provider, spec["handler"], None)), spec["handler"]


def test_no_orphan_action_handlers(provider):
    """Every ``_act_*`` method must be reachable from exactly one tool spec."""
    reachable = [name for spec in _TOOL_SPECS.values() for name in (spec.get("actions") or {}).values()]
    reachable += [spec["handler"] for spec in _TOOL_SPECS.values() if not spec.get("actions")]
    assert len(reachable) == len(set(reachable)), f"handler registered twice: {reachable}"

    declared = {
        name for name, _ in inspect.getmembers(provider, predicate=inspect.ismethod)
        if name.startswith("_act_")
    }
    assert declared == set(reachable), f"orphan or missing handlers: {declared ^ set(reachable)}"


def test_schema_advertises_every_action_handle_tool_call_accepts():
    """The advertised surface and the accepted surface must be identical."""
    assert set(LLM_ACTIONS) == APPROVED_SURFACE

    advertised = set()
    for schema in TOOL_SCHEMAS:
        enum = schema["parameters"]["properties"].get("action", {}).get("enum")
        if enum:
            advertised.update((schema["name"], action) for action in enum)
        else:
            advertised.add((schema["name"], None))
    assert advertised == APPROVED_SURFACE


class _TracingArgs(dict):
    """Dict that records every key a handler asks for."""

    def __init__(self, initial=None):
        super().__init__(initial or {})
        self.touched = set()

    def get(self, key, default=None):
        self.touched.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.touched.add(key)
        return super().__getitem__(key)


def test_every_parameter_a_handler_reads_is_declared(provider):
    """A key missing from the schema never reaches the handler (B10 defect class)."""
    ok(provider, "rekal_remember", "store", content="The deployment uses blue-green releases")
    memory_id = ok(provider, "rekal_search", "search", query="deployment")["results"][0]["id"]

    seeds = {
        "rekal_search": {"action": "search", "query": "deployment", "memory_id": memory_id,
                         "limit": 3, "project": "p", "memory_type": "fact", "fusion": "rrf",
                         "graph_expand": True, "temporal": True, "strategy_boosts": {}},
        "rekal_remember": {"action": "store", "content": "seed fact for tracing",
                           "memory_id": memory_id, "memory_type": "fact", "tags": ["t"],
                           "importance": 0.6, "project": "p", "wing": "w", "room": "r"},
        "rekal_link": {"from_id": memory_id, "to_id": memory_id, "link_relation": "related_to"},
        "rekal_overview": {"action": "timeline", "project": "p", "start": "2000-01-01",
                           "end": "2999-01-01", "limit": 5, "statement": "seed fact"},
        "rekal_ingest": {"turns": ["The release checklist requires a rollback plan before deploy"],
                         "content": "text", "project": "p", "wing": "w", "room": "r"},
        "rekal_diary": {"action": "write", "entry": "traced diary entry", "topic": "t",
                        "limit": 3, "wing": "w", "agent_name": "hermes"},
    }

    for tool_name, spec in _TOOL_SPECS.items():
        declared = set(schema_for(tool_name)["parameters"]["properties"])
        actions = list(spec.get("actions") or (None,))
        for action in actions:
            seed = dict(seeds[tool_name])
            if action is not None:
                seed["action"] = action
            tracing = _TracingArgs(seed)
            try:  # handlers legitimately fail with sentinel values; the reads are the point
                provider.handle_tool_call(tool_name, tracing)
            except Exception:  # pragma: no cover - handle_tool_call contains failures
                pass
            undeclared = tracing.touched - declared
            assert not undeclared, f"{tool_name}/{action} reads undeclared params: {sorted(undeclared)}"


# ── Surface safety ───────────────────────────────────────────────────────

def test_no_destructive_action_on_the_llm_surface():
    for tool_name, spec in _TOOL_SPECS.items():
        for action in (spec.get("actions") or {}):
            offenders = [verb for verb in _DESTRUCTIVE_VERBS if verb in action.lower()]
            assert not offenders, f"{tool_name}/{action} looks destructive ({offenders})"


def test_internal_operations_are_not_addressable_from_the_tool_surface():
    internal_handlers = set(_INTERNAL_OPERATIONS.values())
    dispatch_handlers = {
        handler for spec in _TOOL_SPECS.values()
        for handler in ((spec.get("actions") or {}).values() if spec.get("actions") else [spec["handler"]])
    }
    assert not (internal_handlers & dispatch_handlers)

    all_actions = {action for spec in _TOOL_SPECS.values() for action in (spec.get("actions") or {})}
    assert not (all_actions & EXCLUDED_OPERATIONS)


@pytest.mark.parametrize("operation", sorted(EXCLUDED_OPERATIONS))
def test_excluded_operations_are_unreachable_through_handle_tool_call(provider, operation):
    memory_id = ok(provider, "rekal_remember", "store", content="Deleting this must be operator-only")["memory_id"]
    before = ok(provider, "rekal_overview", "stats")["total_memories"]

    for tool_name in _TOOL_SPECS:
        # Each tool: as the action, and as a bare operation name smuggled in.
        for payload in ({"action": operation}, {"operation": operation, "memory_id": memory_id}):
            result = provider.handle_tool_call(tool_name, dict(payload))
            assert "error" in json.loads(result), (tool_name, payload, result)

    assert ok(provider, "rekal_overview", "stats")["total_memories"] == before
    assert ok(provider, "rekal_search", "get", memory_id=memory_id)["id"] == memory_id


def test_operation_kwarg_is_not_honoured(provider):
    """The old mega-tool took {"operation": ...}; that key must not route anything."""
    result = provider.handle_tool_call("rekal_search", {"operation": "memory_delete"})
    assert "error" in json.loads(result)


def test_schema_handler_drift_fails_closed(provider, monkeypatch):
    monkeypatch.setitem(_TOOL_SPECS["rekal_link"], "handler", "_act_does_not_exist")
    payload = json.loads(call(provider, "rekal_link", from_id="a", to_id="b"))
    assert "error" in payload and "unavailable" in payload["error"]


def test_unknown_tool_and_action_are_rejected(provider):
    assert "Unknown tool" in provider.handle_tool_call("mempalace", {"operation": "search"})
    assert "Unknown tool" in provider.handle_tool_call("definitely_not_a_tool", {})
    assert "Unknown action" in call(provider, "rekal_search", "definitely_not_an_action")
    assert "Unknown action" in call(provider, "rekal_search", None)


# ── Engine-backed actions (D13: thin ports, no storage reimplementation) ──

def test_store_search_recall_and_stats(provider):
    stored = ok(provider, "rekal_remember", "store",
                content="The gateway uses mutual TLS", memory_type="fact")
    assert stored["success"] is True and stored["memory_id"]

    for action in ("search", "recall"):
        payload = ok(provider, "rekal_search", action, query="gateway mutual TLS")
        assert payload["count"] >= 1
        assert "mutual TLS" in payload["results"][0]["content"]
    assert ok(provider, "rekal_search", "recall", query="gateway")["retrieval"]["mode"] == "rrf"

    stats = ok(provider, "rekal_overview", "stats")
    assert stats["total_memories"] >= 1


def test_search_requires_query(provider):
    assert "query is required" in call(provider, "rekal_search", "search")


def test_get_returns_memory_with_links(provider):
    first = ok(provider, "rekal_remember", "store", content="The parser handles nested expressions")["memory_id"]
    second = ok(provider, "rekal_remember", "store", content="The parser rejects unbalanced parens")["memory_id"]
    ok(provider, "rekal_link", from_id=first, to_id=second, link_relation="related_to")

    fetched = ok(provider, "rekal_search", "get", memory_id=first)
    assert fetched["id"] == first
    assert [link["id"] for link in fetched["links"]] == [second]
    assert "not found" in call(provider, "rekal_search", "get", memory_id="nope")


def test_reinforce_increments_proof_count(provider):
    memory_id = ok(provider, "rekal_remember", "store", content="The build uses Ninja")["memory_id"]
    first = ok(provider, "rekal_remember", "reinforce", memory_id=memory_id)["proof_count"]
    second = ok(provider, "rekal_remember", "reinforce", memory_id=memory_id)["proof_count"]
    assert second == first + 1
    assert "not found" in call(provider, "rekal_remember", "reinforce", memory_id="missing")


def test_duplicate_store_reinforces_instead_of_duplicating(provider):
    ok(provider, "rekal_remember", "store", content="Caching is enabled for the session store")
    again = ok(provider, "rekal_remember", "store", content="Caching is enabled for the session store")
    assert again.get("duplicate") is True
    assert again.get("reinforced") is True
    assert ok(provider, "rekal_overview", "stats")["total_memories"] == 1


def test_store_validates_memory_type_and_importance(provider):
    assert "memory_type must be one of" in call(provider, "rekal_remember", "store",
                                                content="x", memory_type="nonsense")
    assert "content is required" in call(provider, "rekal_remember", "store", content="   ")
    stored = ok(provider, "rekal_remember", "store", content="Overweight importance", importance=9)
    assert stored["importance"] == 1.0


def test_link_validates_relation(provider):
    a = ok(provider, "rekal_remember", "store", content="Fact about alpha")["memory_id"]
    b = ok(provider, "rekal_remember", "store", content="Fact about beta")["memory_id"]
    assert ok(provider, "rekal_link", from_id=a, to_id=b)["relation"] == "related_to"
    assert "link_relation must be one of" in call(provider, "rekal_link", from_id=a, to_id=b,
                                                  link_relation="deletes")
    assert "from_id is required" in call(provider, "rekal_link", to_id=b)


def test_timeline_and_topics(provider):
    ok(provider, "rekal_remember", "store", content="The release train leaves on Tuesdays", memory_type="fact")
    ok(provider, "rekal_remember", "store", content="Prefer short commit messages", memory_type="preference")

    timeline = ok(provider, "rekal_overview", "timeline", limit=5)
    assert timeline["count"] == 2
    topics = ok(provider, "rekal_overview", "topics")["topics"]
    assert {topic["topic"] for topic in topics} == {"fact", "preference"}


def test_contradictions_lists_pairs_and_checks_statements(provider):
    ok(provider, "rekal_remember", "store", content="Caching is enabled for the session store", project="p")
    ok(provider, "rekal_remember", "store", content="Caching is disabled for the session store", project="p")

    conflicts = ok(provider, "rekal_overview", "contradictions", project="p")
    assert conflicts["count"] >= 1

    check = ok(provider, "rekal_overview", "contradictions",
               statement="Caching is disabled for the session store")
    assert check["verdict"] != "unverified"
    assert "statement" in check


def test_ingest_accepts_string_list(provider):
    payload = ok(provider, "rekal_ingest",
                 turns=["The release checklist requires a rollback plan before deploy"])
    assert payload["success"] is True and payload["stored"] == 1
    assert "turns is required" in call(provider, "rekal_ingest")


def test_engine_actions_fail_closed_when_uninitialized():
    p = RekalMemoryProvider()
    assert "not initialized" in p.handle_tool_call("rekal_search", {"action": "search", "query": "x"})
    assert "not initialized" in p.internal_op("memory_delete", memory_id="x")


# ── C1: store-backed diary (no mempalace required) ───────────────────────

def test_diary_round_trip_without_mempalace(provider):
    written = ok(provider, "rekal_diary", "write",
                 entry="Fixed the WAL checkpoint bug", topic="work")
    assert written["success"] is True
    assert written["entry_id"] and written["agent"] == "hermes" and written["timestamp"]

    entries = ok(provider, "rekal_diary", "read")["entries"]
    assert [entry["content"] for entry in entries] == ["Fixed the WAL checkpoint bug"]
    assert entries[0]["topic"] == "work"
    assert set(entries[0]) >= {"timestamp", "topic", "content"}


def test_diary_is_durable_across_provider_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(rekal_plugin, "_load_mempalace", lambda: None)
    monkeypatch.setattr(rekal_plugin, "_mcp_mod", None)
    monkeypatch.setattr(rekal_plugin, "_MEMPALACE_IMPORT_ERROR", "not installed")

    first = RekalMemoryProvider()
    first.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    ok(first, "rekal_diary", "write", entry="Session one summary")
    first.shutdown()

    second = RekalMemoryProvider()
    second.initialize("s2", hermes_home=str(tmp_path), platform="cli")
    try:
        entries = ok(second, "rekal_diary", "read")["entries"]
        assert [entry["content"] for entry in entries] == ["Session one summary"]
    finally:
        second.shutdown()


def test_diary_read_is_newest_first_and_limited(provider):
    for index in range(3):
        ok(provider, "rekal_diary", "write", entry=f"entry {index}", topic="t")
    entries = ok(provider, "rekal_diary", "read", limit=2)["entries"]
    assert len(entries) == 2
    assert entries[0]["content"] == "entry 2"


def test_diary_read_is_agent_scoped(provider):
    ok(provider, "rekal_diary", "write", entry="hermes note", agent_name="hermes")
    ok(provider, "rekal_diary", "write", entry="other note", agent_name="other-agent")
    contents = [entry["content"] for entry in ok(provider, "rekal_diary", "read")["entries"]]
    assert contents == ["hermes note"]


def test_diary_entries_stay_out_of_memory_search(provider):
    """A journal line must not become a recall candidate or a conflict source."""
    ok(provider, "rekal_diary", "write", entry="Kaleidoscope journal line about nothing durable")
    payload = ok(provider, "rekal_search", "recall", query="kaleidoscope journal")
    assert payload["count"] == 0


def test_diary_requires_an_entry(provider):
    assert "entry is required" in call(provider, "rekal_diary", "write", entry="")


def test_internal_diary_ops_keep_the_legacy_call_shape(provider):
    written = json.loads(provider.internal_op("diary_write", entry="legacy shape", topic="ops"))
    assert set(written) >= {"entry_id", "agent", "timestamp"}
    read = json.loads(provider.internal_op("diary_read", limit=5))
    assert read["entries"][0]["content"] == "legacy shape"


# ── Internal / CLI-only capability preservation ──────────────────────────

def test_internal_op_preserves_destructive_capability(provider):
    memory_id = ok(provider, "rekal_remember", "store", content="Operator-only deletion target")["memory_id"]
    result = json.loads(provider.internal_op("memory_delete", memory_id=memory_id))
    assert result.get("success") is True
    assert "not found" in call(provider, "rekal_search", "get", memory_id=memory_id)


def test_internal_op_keeps_read_capabilities_off_the_tool_surface(provider):
    ok(provider, "rekal_remember", "store", content="The scheduler retries on 503")
    payload = json.loads(provider.internal_op("build_context", query="scheduler"))
    assert isinstance(payload["memories"], list)
    assert json.loads(provider.internal_op("memory_health"))["total_memories"] == 1


def test_internal_consolidate_and_maintenance_are_dry_run_and_idempotent(provider):
    ok(provider, "rekal_remember", "store", content="duplicate-prone row")
    assert json.loads(provider.internal_op("consolidate"))["dry_run"] is True
    assert json.loads(provider.internal_op("migrate"))["success"] is True
    assert json.loads(provider.internal_op("rebuild_fts"))["success"] is True
    assert json.loads(provider.internal_op("rebuild_fts"))["success"] is True


def test_internal_op_rejects_unknown_operation(provider):
    assert "Unknown internal operation" in provider.internal_op("rm_rf")
    assert "memory_delete" in provider.internal_operations()


def test_internal_set_config_keeps_scoring_weights_operator_only(provider):
    ok(provider, "rekal_remember", "store", content="Weighted row", project="p")
    assert json.loads(provider.internal_op("set_config", project="p", key="w_fts", value=2.0))["success"] is True
    actions = {action for spec in _TOOL_SPECS.values() for action in (spec.get("actions") or {})}
    assert "set_config" not in actions


# ── Lifecycle ────────────────────────────────────────────────────────────

def test_prefetch_injects_mempalace_context(provider):
    ok(provider, "rekal_remember", "store", content="The deployment uses blue-green releases")
    ctx = provider.prefetch("deployment releases")
    assert ctx.startswith("<mempalace_context>")
    assert ctx.endswith("</mempalace_context>")
    assert "blue-green" in ctx


def test_capture_and_session_end_ingest(provider):
    provider.sync_turn("I prefer concise answers", "Understood.")
    provider.on_session_end([])
    assert ok(provider, "rekal_overview", "stats")["total_memories"] >= 1


def test_on_memory_write_mirrors_add(provider):
    provider.on_memory_write("add", "memory", "The user works in the hermes-agent repository")
    assert "hermes-agent" in provider.prefetch("repository")


def test_system_prompt_block_names_the_scoped_tools(provider):
    block = provider.system_prompt_block()
    assert "<memory_system>" in block
    assert "<mempalace_context>" in block
    assert "rekal_search" in block and "rekal_remember" in block
    # The prompt must not advertise operations the surface no longer has.
    for stale in ("session_init", "mempalace tool", "set_config", "memory_delete"):
        assert stale not in block


def test_backup_paths(provider, tmp_path):
    assert provider.backup_paths() == [str(tmp_path / "rekal" / "rekal_memories.db")]


def test_auto_recall_can_be_disabled(tmp_path, monkeypatch):
    (tmp_path / "rekal.json").write_text(json.dumps({"auto_recall": False}), encoding="utf-8")
    monkeypatch.setattr(rekal_plugin, "_load_mempalace", lambda: None)
    p = RekalMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    try:
        ok(p, "rekal_remember", "store", content="Some durable fact about the system")
        assert p.prefetch("durable fact") == ""
    finally:
        p.shutdown()


def test_status_config_reports_counts(provider):
    ok(provider, "rekal_remember", "store", content="Status line fixture")
    summary = provider.get_status_config({})["summary"]
    assert "memories" in summary and "links" in summary


def test_registration(monkeypatch):
    registered = []

    class _Ctx:
        def register_memory_provider(self, provider):
            registered.append(provider)

    rekal_plugin.register(_Ctx())
    assert len(registered) == 1
    assert registered[0].name == "rekal"
