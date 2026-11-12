"""B7 — the enhanced backend honours ``agent_context``.

``MemoryManager.initialize_all`` already plumbs ``agent_context`` ("primary" | "cron" |
"subagent" | "flush", per the ``MemoryProvider.initialize`` contract), but
``EnhancedMemoryBackend.initialize`` swallowed ``**kwargs``, so cron and delegated-child
turns were mined into the PRIMARY profile store. Rekal resolves the same flag
(``plugins/memory/rekal/__init__.py``: ``agent_context not in {"cron", "flush",
"subagent"}``); the enhanced layer now mirrors it.
"""

import pytest

from agent.memory.backend import EnhancedMemoryBackend
from agent.memory_manager import MemoryManager

PREFERENCE = "I prefer dark mode in every project"


def _backend(tmp_path, agent_context):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("sess", agent_context=agent_context, platform="cli")
    return backend


@pytest.mark.parametrize("agent_context", ["cron", "subagent", "flush"])
def test_non_interactive_context_does_not_capture(tmp_path, agent_context):
    backend = _backend(tmp_path, agent_context)
    assert backend.capture_enabled is False

    backend.observe_turn(PREFERENCE, "Noted.")
    backend.on_session_end([{"role": "user", "content": PREFERENCE}])
    backend.on_pre_compress([{"role": "user", "content": PREFERENCE}])
    backend.on_memory_write("add", "user", PREFERENCE)

    assert backend.health()["total_memories"] == 0
    backend.shutdown()


def test_primary_context_still_captures(tmp_path):
    """Control: the gate must not be a blanket capture kill."""
    backend = _backend(tmp_path, "primary")
    assert backend.capture_enabled is True

    backend.observe_turn(PREFERENCE, "Noted.")

    assert backend.health()["total_memories"] >= 1
    backend.shutdown()


def test_missing_agent_context_keeps_capture(tmp_path):
    """Providers written before the contract pass no ``agent_context``; they keep working."""
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("sess")

    assert backend.capture_enabled is True
    backend.observe_turn(PREFERENCE, "Noted.")
    assert backend.health()["total_memories"] >= 1
    backend.shutdown()


def test_non_interactive_context_still_recalls(tmp_path):
    """Skip writes, not recall: a cron turn may still read what the user already has."""
    writer = _backend(tmp_path, "primary")
    writer.remember(PREFERENCE)
    writer.shutdown()

    reader = _backend(tmp_path, "cron")
    assert "dark mode" in reader.prefetch("dark mode")
    reader.shutdown()


def test_manager_plumbing_reaches_the_backend(tmp_path):
    """End-to-end: the kwarg ``agent_init`` builds is the one the store obeys."""
    manager = MemoryManager(enhanced_config={"enabled": True})
    manager.initialize_all(session_id="sess", hermes_home=str(tmp_path),
                           platform="cron", agent_context="cron")
    assert manager.enhanced_enabled is True

    manager.sync_all(PREFERENCE, "Noted.", session_id="sess")
    assert manager.flush_pending(timeout=5) is True

    assert manager._enhanced.health()["total_memories"] == 0
    manager.shutdown_all()


def test_manager_primary_context_captures(tmp_path):
    manager = MemoryManager(enhanced_config={"enabled": True})
    manager.initialize_all(session_id="sess", hermes_home=str(tmp_path),
                           platform="cli", agent_context="primary")

    manager.sync_all(PREFERENCE, "Noted.", session_id="sess")
    assert manager.flush_pending(timeout=5) is True

    assert manager._enhanced.health()["total_memories"] >= 1
    manager.shutdown_all()
