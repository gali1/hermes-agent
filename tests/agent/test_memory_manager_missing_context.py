"""D11 / C7 — missing-context detection at the turn-start recall surface.

D11 ("prompt build") and C7 ("turn start") are the SAME code path here: the system
prompt carries only static guidance, and every recall runs at turn start in
``agent/turn_context._memory_turn_start_and_prefetch`` → ``MemoryManager.prefetch_all``.
So the marker is emitted once, in ``prefetch_all``; these tests cover the manager surface
(D11) and the real turn-start entry point (C7), and prove it does not double-fire.
"""

from types import SimpleNamespace

from agent.memory_manager import (
    MISSING_CONTEXT_MARKER,
    MemoryManager,
    build_memory_context_block,
    references_prior_work,
)
from agent.turn_context import _memory_turn_start_and_prefetch

PRIOR_WORK_QUERY = "as we discussed, continue with the migration you started"


class EmptyProvider:
    """A provider that was queried and had nothing: recall happened, recall was empty."""

    name = "empty"

    def get_tool_schemas(self):
        return []

    def prefetch(self, query, *, session_id=""):
        self.calls = getattr(self, "calls", 0) + 1
        return ""

    def sync_turn(self, *args, **kwargs):
        return None


class HitProvider(EmptyProvider):
    name = "hit"

    def prefetch(self, query, *, session_id=""):
        return "- the user prefers dark mode"


def _manager_with(provider=None, enhanced=False, tmp_path=None):
    manager = MemoryManager(enhanced_config={"enabled": enhanced})
    if provider is not None:
        manager.add_provider(provider)
    if enhanced:
        manager.initialize_all(session_id="s", hermes_home=str(tmp_path))
    return manager


def test_marker_fires_when_prior_work_is_referenced_but_recall_is_empty(tmp_path):
    manager = _manager_with(EmptyProvider())
    assert manager.prefetch_all(PRIOR_WORK_QUERY) == MISSING_CONTEXT_MARKER
    assert "session_search" in MISSING_CONTEXT_MARKER
    assert "Do not invent" in MISSING_CONTEXT_MARKER


def test_no_marker_on_a_turn_that_does_not_reference_prior_work(tmp_path):
    manager = _manager_with(EmptyProvider())
    assert manager.prefetch_all("write a bash loop over the files") == ""


def test_no_marker_when_recall_returned_something(tmp_path):
    manager = _manager_with(HitProvider())
    ctx = manager.prefetch_all(PRIOR_WORK_QUERY)
    assert "dark mode" in ctx
    assert MISSING_CONTEXT_MARKER not in ctx


def test_no_marker_when_there_is_no_recall_source_at_all(tmp_path):
    """With no provider and the enhanced layer off, nothing was consulted: stay silent."""
    manager = MemoryManager()
    assert manager.prefetch_all(PRIOR_WORK_QUERY) == ""


def test_enhanced_recall_counts_as_a_recall_source(tmp_path):
    manager = _manager_with(enhanced=True, tmp_path=tmp_path)
    assert manager.enhanced_enabled is True
    assert manager.prefetch_all(PRIOR_WORK_QUERY) == MISSING_CONTEXT_MARKER


def test_detector_is_bounded_and_cheap():
    """The cue is scanned from the head of the turn only — no full-text walk, no store I/O."""
    assert references_prior_work("as we discussed, the parser fix matters")
    assert references_prior_work("Continue where we left off")
    assert not references_prior_work("hello")
    assert not references_prior_work("")
    # A cue buried past the scan window does not cost a full scan.
    buried = "x " * 2500 + "as we discussed"
    assert len(buried) > 4000
    assert not references_prior_work(buried)


def _fake_agent(manager):
    return SimpleNamespace(
        _memory_manager=manager,
        _user_turn_count=2,
        session_id="s",
        _emit_status=lambda *a, **k: None,
    )


def test_turn_start_surface_emits_the_marker_once(tmp_path):
    """C7: the real turn-start entry point returns the marker, and only once."""
    manager = _manager_with(EmptyProvider())
    cache = _memory_turn_start_and_prefetch(_fake_agent(manager), PRIOR_WORK_QUERY)
    assert cache == MISSING_CONTEXT_MARKER
    fenced = build_memory_context_block(cache)
    assert fenced.count("Missing context:") == 1
    assert "<memory-context>" in fenced


def test_turn_start_surface_stays_silent_without_a_recall_source():
    cache = _memory_turn_start_and_prefetch(_fake_agent(MemoryManager()), PRIOR_WORK_QUERY)
    assert cache == ""
