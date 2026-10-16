"""Coexistence contract: enhanced memory is on by default and everything else
that already ran keeps running alongside it.

The enhanced local layer (``memory.enhanced.enabled: true`` in DEFAULT_CONFIG)
must not disable, replace or alter:

* built-in curated memory (``MEMORY.md`` / ``USER.md`` and the ``memory``
  tool's add/replace/remove/batch),
* conversation history (SessionDB under ``state.db``),
* content-aware tool-output compression (``compression.content_aware``),
* session continuation across reboot (``session_continuation.json``),
* external memory provider plugins (opt-in via ``memory.provider``).
"""

import json

from hermes_cli.config_defaults import DEFAULT_CONFIG


class _FakeProvider:
    name = "fake"

    def __init__(self):
        self.prefetched = 0
        self.synced = []
        self.closed = False

    def get_tool_schemas(self):
        return []

    def prefetch(self, query, *, session_id=""):
        self.prefetched += 1
        return "provider recall"

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self.synced.append((user_content, assistant_content))

    def on_session_end(self, messages):
        pass

    def shutdown(self):
        self.closed = True


def test_enhanced_memory_is_enabled_by_default():
    assert DEFAULT_CONFIG["memory"]["enhanced"]["enabled"] is True
    # External providers remain opt-in; the enhanced layer is built-in only.
    assert DEFAULT_CONFIG["memory"]["provider"] == ""


def test_builtin_curated_memory_unaffected_while_enhanced_active(tmp_path):
    from agent.memory.backend import EnhancedMemoryBackend
    from tools.memory_tool import MemoryStore as CuratedStore, memory_tool

    store = CuratedStore()
    store.load_from_disk()
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    try:
        added = json.loads(memory_tool(
            action="add", target="memory", content="A curated fact",
            store=store, backend=backend,
        ))
        assert added["success"] is True

        replaced = json.loads(memory_tool(
            action="replace", target="memory", content="A curated fact v2",
            old_text="A curated fact", store=store, backend=backend,
        ))
        assert replaced["success"] is True

        removed = json.loads(memory_tool(
            action="remove", target="memory", old_text="A curated fact v2",
            store=store, backend=backend,
        ))
        assert removed["success"] is True

        # The enhanced backend stayed available and untouched by the tool ops.
        assert backend.available is True
        assert backend.store is not None
    finally:
        backend.shutdown()


def test_history_continuation_and_memory_paths_are_separate(tmp_path):
    from agent.memory.backend import EnhancedMemoryBackend
    from agent import session_continuation as sc

    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    try:
        enhanced_db = backend.db_path
    finally:
        backend.shutdown()

    continuation_path = sc.status(home=str(tmp_path))["path"]
    session_db = str(tmp_path / "state.db")

    paths = {enhanced_db, continuation_path, session_db}
    assert len(paths) == 3, paths
    assert enhanced_db.endswith("memory_engine/memories.db")
    assert continuation_path.endswith("session_continuation.json")


def test_compression_settings_independent_of_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.content_compression.config import get_content_compression_config, reset_cache

    reset_cache()
    config = get_content_compression_config()
    assert config["enabled"] is True
    assert config["lossless_folds"] is True
    # No memory keys leaked into the compression namespace.
    assert "enhanced" not in config


def test_provider_plugin_path_coexists_with_enhanced(tmp_path):
    from agent.memory_manager import MemoryManager

    provider = _FakeProvider()
    manager = MemoryManager(enhanced_config={"enabled": True})
    manager.add_provider(provider)
    manager.initialize_all(session_id="s", hermes_home=str(tmp_path))
    try:
        assert manager.enhanced_enabled is True
        # Provider recall still flows; the empty enhanced store adds nothing.
        assert manager.prefetch_all("anything") == "provider recall"
        manager.sync_all("user text", "assistant text")
        assert manager.flush_pending(timeout=5) is True
        assert provider.synced == [("user text", "assistant text")]
    finally:
        manager.shutdown_all()
    assert provider.closed is True
