"""MemoryManager integration tests for the additive enhanced layer.

These pin the contract that the enhanced layer is enabled by default
(``memory.enhanced.enabled: true`` in DEFAULT_CONFIG) with an explicit
opt-out, coexists with external providers, and leaves the provider-only path
byte-identical when disabled.
"""

import threading
import time

from agent.memory_manager import MemoryManager


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.synced = []
        self.prefetch_calls = 0
        self.ended = False
        self.closed = False
        self.system_block = "FAKE PROVIDER MEMORY"

    def get_tool_schemas(self):
        return []

    def system_prompt_block(self):
        return self.system_block

    def prefetch(self, query, *, session_id=""):
        self.prefetch_calls += 1
        return "provider recall"

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self.synced.append((user_content, assistant_content))

    def on_session_end(self, messages):
        self.ended = True

    def shutdown(self):
        self.closed = True


def test_disabled_manager_behavior_unchanged(tmp_path):
    manager = MemoryManager()
    provider = FakeProvider()
    manager.add_provider(provider)
    manager.initialize_all(session_id="s", hermes_home=str(tmp_path))

    assert manager.enhanced_enabled is False
    assert manager.enhanced_requested is False
    assert manager.prefetch_all("hello") == "provider recall"
    assert manager.build_system_prompt() == "FAKE PROVIDER MEMORY"

    manager.sync_all("user text", "assistant text")
    assert manager.flush_pending(timeout=5) is True
    assert provider.synced == [("user text", "assistant text")]

    manager.shutdown_all()
    assert provider.closed is True


def test_enhanced_only_manager(tmp_path):
    manager = MemoryManager(enhanced_config={"enabled": True, "auto_recall": True})
    manager.initialize_all(session_id="s", hermes_home=str(tmp_path), platform="cli")

    assert manager.enhanced_requested is True
    assert manager.enhanced_enabled is True
    assert manager.prefetch_all("hello") == ""

    manager.sync_all("I prefer dark mode in every project", "Noted.")
    assert manager.flush_pending(timeout=5) is True

    ctx = manager.prefetch_all("dark mode")
    assert "dark mode" in ctx

    manager.on_memory_write("add", "user", "The user prefers dark mode everywhere")
    assert manager._enhanced.health()["total_memories"] >= 2

    assert manager.on_pre_compress([]) == ""
    manager.shutdown_all()
    assert manager.enhanced_enabled is False


def test_enhanced_and_provider_coexist(tmp_path):
    manager = MemoryManager(enhanced_config={"enabled": True})
    provider = FakeProvider()
    manager.add_provider(provider)
    manager.initialize_all(session_id="s", hermes_home=str(tmp_path))

    manager.sync_all("We decided to use PostgreSQL 17 for the database", "ok")
    assert manager.flush_pending(timeout=5) is True
    assert provider.synced

    ctx = manager.prefetch_all("PostgreSQL")
    assert "provider recall" in ctx
    assert "PostgreSQL" in ctx

    manager.on_session_end([])
    assert provider.ended is True
    manager.shutdown_all()
    assert provider.closed is True


def test_enhanced_failure_degrades_to_builtin(tmp_path, monkeypatch):
    import agent.memory.backend as backend_module

    def _boom(path):
        raise RuntimeError("no store")

    monkeypatch.setattr(backend_module, "MemoryStore", _boom)
    manager = MemoryManager(enhanced_config={"enabled": True})
    manager.initialize_all(session_id="s", hermes_home=str(tmp_path))

    assert manager.enhanced_requested is True
    assert manager.enhanced_enabled is False
    assert manager.prefetch_all("hello") == ""
    manager.sync_all("a durable fact", "ok")
    assert manager.flush_pending(timeout=5) is True
    manager.shutdown_all()


def test_enhanced_session_switch_updates_session(tmp_path):
    manager = MemoryManager(enhanced_config={"enabled": True})
    manager.initialize_all(session_id="s1", hermes_home=str(tmp_path))
    assert manager._enhanced._session_id == "s1"

    manager.on_session_switch("s2", parent_session_id="s1")
    assert manager._enhanced._session_id == "s2"
    manager.shutdown_all()


# ---------------------------------------------------------------------------
# B8: the enhanced prefetch must be bounded exactly like an external provider.
# Before this, it ran inline: a slow FTS5 query or a cold embedding model
# stalled the whole turn with no ceiling, while plugin providers had an 8s cap.
# ---------------------------------------------------------------------------

class _StubEnhanced:
    """Minimal stand-in for EnhancedMemoryBackend: only prefetch is exercised."""

    available = True          # MemoryManager.enhanced_enabled requires this

    def __init__(self, *, delay=0.0, payload="enhanced recall", gate=None):
        self.delay = delay
        self.payload = payload
        self.gate = gate
        self.calls = 0

    def prefetch(self, query, *, session_id=""):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        elif self.delay:
            threading.Event().wait(self.delay)
        return self.payload

    def health(self):
        return {}


def _manager_with(stub, timeout=0.2):
    manager = MemoryManager(external_prefetch_timeout=timeout,
                            enhanced_config={"enabled": True})
    manager._enhanced = stub
    return manager


def test_enhanced_prefetch_is_bounded_by_timeout():
    stub = _StubEnhanced(delay=5.0)
    manager = _manager_with(stub, timeout=0.2)

    started = time.monotonic()
    ctx = manager.prefetch_all("hello")
    elapsed = time.monotonic() - started

    assert ctx == ""                                   # stuck prefetch contributes nothing
    assert elapsed < 2.0                               # and the turn is not held open
    assert stub.calls == 1


def test_enhanced_prefetch_stuck_call_is_skipped_on_later_turns():
    gate = threading.Event()
    stub = _StubEnhanced(gate=gate)
    manager = _manager_with(stub, timeout=0.1)

    assert manager.prefetch_all("first") == ""
    assert manager.prefetch_all("second") == ""         # still running -> skipped, not stacked
    assert stub.calls == 1                              # no second call was started

    gate.set()                                          # let the daemon thread finish
    time.sleep(0.2)


def test_enhanced_prefetch_content_is_unchanged_when_fast():
    stub = _StubEnhanced(payload="ENHANCED BLOCK")
    manager = _manager_with(stub, timeout=5.0)

    assert manager.prefetch_all("hello") == "ENHANCED BLOCK"
    assert stub.calls == 1


def test_enhanced_prefetch_error_is_non_fatal():
    class Boom(_StubEnhanced):
        def prefetch(self, query, *, session_id=""):
            raise RuntimeError("store gone")

    manager = _manager_with(Boom())
    assert manager.prefetch_all("hello") == ""          # swallowed, turn continues
