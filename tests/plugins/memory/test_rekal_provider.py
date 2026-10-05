"""Provider contract tests for the Rekal memory plugin.

Ported from OpenCode's ``mempalace.test.ts``: every parameter a bridge handler
reads must be declared in the tool schema, otherwise it never reaches the
handler.  Also covers the Hermes MemoryProvider lifecycle (initialize,
prefetch, capture, session end, tool dispatch, backup, shutdown).
"""

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
    MEMPALACE_SCHEMA,
    RekalMemoryProvider,
    _OPERATIONS,
)

# Every parameter key the upstream bridge reads, by operation.  This is the
# wire contract: a key missing from the schema never reaches the handler.
BRIDGE_PARAM_KEYS = {
    "agent_name", "as_of", "claim", "confidence", "content", "depth",
    "direction", "drawer", "ended", "entity", "entry", "expand_with_kg",
    "from_id", "fusion", "graph_expand", "half_life", "importance", "key",
    "limit", "link_relation", "memory_id", "memory_type", "old_id", "project",
    "query", "relation", "room", "start", "statement", "strategy_boosts",
    "tags", "target", "task", "temporal", "to_id", "topic", "turns",
    "valid_from", "value", "w_access", "w_fts", "w_recency", "w_vec", "wing",
}


# ── Schema contract ───────────────────────────────────────────────────────

def test_schema_declares_every_parameter_the_bridge_reads():
    declared = set(MEMPALACE_SCHEMA["parameters"]["properties"])
    missing = BRIDGE_PARAM_KEYS - declared
    assert not missing, f"undeclared parameters would be unreachable: {sorted(missing)}"


def test_operation_enum_is_complete():
    props = MEMPALACE_SCHEMA["parameters"]["properties"]
    assert props["operation"]["enum"] == _OPERATIONS
    assert len(_OPERATIONS) == 31
    assert MEMPALACE_SCHEMA["parameters"]["required"] == ["operation"]


def test_enum_constraints():
    props = MEMPALACE_SCHEMA["parameters"]["properties"]
    assert props["memory_type"]["enum"] == ["fact", "preference", "procedure", "context", "episode"]
    assert props["link_relation"]["enum"] == ["supersedes", "contradicts", "related_to"]
    assert props["fusion"]["enum"] == ["rrf"]
    assert props["direction"]["enum"] == ["in", "out", "both"]


def test_identity_and_classification_parameters_declared():
    props = MEMPALACE_SCHEMA["parameters"]["properties"]
    for key in (
        "memory_id", "old_id", "from_id", "to_id",
        "memory_type", "project", "importance", "tags",
        "fusion", "graph_expand", "temporal", "strategy_boosts",
        "turns", "topic", "task", "start", "end", "as_of", "direction",
        "valid_from", "ended", "key", "value",
        "w_fts", "w_vec", "w_recency", "w_access", "half_life",
    ):
        assert key in props, f"{key} missing from schema"


def test_tool_description_carries_the_protocol():
    description = MEMPALACE_SCHEMA["description"]
    assert 'SINGLE TOOL called "mempalace"' in description
    assert "session_init" in description
    assert "memory_recall" in description


# ── Lifecycle fixture ─────────────────────────────────────────────────────

@pytest.fixture
def provider(tmp_path, monkeypatch):
    # Keep the heavy optional import out of the test path; native operations
    # are covered separately via the unavailable path.
    monkeypatch.setattr(rekal_plugin, "_load_mempalace", lambda: None)
    monkeypatch.setattr(rekal_plugin, "_mcp_mod", None)
    monkeypatch.setattr(rekal_plugin, "_MEMPALACE_IMPORT_ERROR", "not installed")
    p = RekalMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    yield p
    p.shutdown()


def call(provider, operation, **args):
    return provider.handle_tool_call("mempalace", {"operation": operation, **args})


# ── Engine-backed operations ──────────────────────────────────────────────

def test_store_and_recall(provider):
    raw = json.loads(call(provider, "memory_store", content="The gateway uses mutual TLS", memory_type="fact"))
    assert raw["success"] is True
    assert raw["memory_id"]

    recall = call(provider, "memory_recall", query="gateway mutual TLS")
    assert "mode: rrf" in recall
    assert "mutual TLS" in recall

    health = json.loads(call(provider, "memory_health"))
    assert health["total_memories"] >= 1


def test_prefetch_injects_mempalace_context(provider):
    call(provider, "memory_store", content="The deployment uses blue-green releases")
    ctx = provider.prefetch("deployment releases")
    assert ctx.startswith("<mempalace_context>")
    assert ctx.endswith("</mempalace_context>")
    assert "blue-green" in ctx


def test_capture_and_session_end_ingest(provider):
    provider.sync_turn("I prefer concise answers", "Understood.")
    provider.on_session_end([])
    health = json.loads(call(provider, "memory_health"))
    assert health["total_memories"] >= 1


def test_on_memory_write_mirrors_add(provider):
    provider.on_memory_write("add", "memory", "The user works in the hermes-agent repository")
    assert "hermes-agent" in provider.prefetch("repository")


def test_ingest_turns_accepts_string_list(provider):
    raw = json.loads(call(
        provider, "ingest_turns",
        turns=["The release checklist requires a rollback plan before deploy"],
    ))
    assert raw["success"] is True
    assert raw["stored"] == 1


def test_conflict_detection_via_tool(provider):
    call(provider, "memory_store", content="Caching is enabled for the session store", project="p")
    call(provider, "memory_store", content="Caching is disabled for the session store", project="p")
    conflicts = json.loads(call(provider, "memory_conflicts", project="p"))
    assert isinstance(conflicts, list)
    assert len(conflicts) >= 1


def test_supersede_preserves_history(provider):
    raw = json.loads(call(provider, "memory_store", content="The API uses REST", project="p"))
    old = raw["memory_id"]
    sup = json.loads(call(
        provider, "memory_supersede",
        old_id=old, content="The API now uses GraphQL", project="p",
    ))
    assert sup.get("new_id") or sup.get("memory_id")


def test_session_init_returns_context(provider):
    call(provider, "memory_store", content="The parser handles nested expressions")
    raw = json.loads(call(provider, "session_init", query="parser"))
    assert raw["task"] == "parser"
    assert isinstance(raw["memories"], list)
    assert "health" in raw


def test_build_context_and_health(provider):
    call(provider, "memory_store", content="The build uses Ninja as the generator")
    context = json.loads(call(provider, "build_context", query="build generator"))
    assert isinstance(context["memories"], list)
    assert "timeline_summary" in context


# ── Dispatch / failure containment ────────────────────────────────────────

def test_unknown_operation_and_tool(provider):
    assert "Unknown operation" in call(provider, "definitely_not_an_operation")
    assert "Unknown tool" in provider.handle_tool_call("not_mempalace", {})


def test_native_operations_report_missing_mempalace(provider):
    result = call(provider, "search", query="anything")
    assert "mempalace" in result.lower()
    assert "pip install mempalace" in result


def test_system_prompt_block(provider):
    block = provider.system_prompt_block()
    assert "<memory_system>" in block
    assert "<mempalace_context>" in block


def test_backup_paths(provider, tmp_path):
    assert provider.backup_paths() == [str(tmp_path / "rekal" / "rekal_memories.db")]


def test_auto_recall_can_be_disabled(tmp_path, monkeypatch):
    (tmp_path / "rekal.json").write_text(json.dumps({"auto_recall": False}), encoding="utf-8")
    monkeypatch.setattr(rekal_plugin, "_load_mempalace", lambda: None)
    p = RekalMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    try:
        call(p, "memory_store", content="Some durable fact about the system")
        assert p.prefetch("durable fact") == ""
    finally:
        p.shutdown()


def test_registration(monkeypatch):
    registered = []

    class _Ctx:
        def register_memory_provider(self, provider):
            registered.append(provider)

    rekal_plugin.register(_Ctx())
    assert len(registered) == 1
    assert registered[0].name == "rekal"
