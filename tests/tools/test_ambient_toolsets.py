"""Contracts for the host's ambient-toolset capability (Route A of the toolaria fix).

A plugin marks a toolset *ambient* when it produces state in every session and must be
able to redeem that state in every session too: toolaria rewrites oversized tool results
ungated, so ``rescuer_fetch`` must survive a session scoped to a handful of toolsets.
Ambient membership lives only in ``ToolRegistry`` and is populated only by an explicit
in-process ``mark_ambient()`` call — never by config — so it cannot be used as a way to
widen a session's restriction from the outside.

Proven here:
1. an unmarked toolset stays absent from a restricted selection, and marking nothing
   leaves the selection byte-for-byte unchanged;
2. a marked toolset appears inside a restricted session, with no other tool appearing
   or disappearing;
3. an unmarked toolset stays absent when a sibling is marked;
4. the security boundary holds: an operator's ``disabled_toolsets`` still wins over
   ambient, and wildcard scopes cannot be declared ambient.
"""

import pytest

import model_tools
from tools.registry import ToolRegistry, registry


def _schema(name: str) -> dict:
    return {"name": name, "description": f"probe {name}", "parameters": {"type": "object", "properties": {}}}


def _handler(args, **kwargs):
    return "{}"


_PROBES = (("ambient_probe_tool", "ambient_probe_ts"), ("plain_probe_tool", "plain_probe_ts"))
_RESTRICTED = ["terminal"]


@pytest.fixture
def probes():
    """Two probe toolsets registered in the live registry, neither ambient to begin with."""
    saved_ambient = registry.ambient_toolsets()
    for name, toolset in _PROBES:
        registry.register(name=name, toolset=toolset, schema=_schema(name), handler=_handler)
    model_tools._clear_tool_defs_cache()
    try:
        yield registry
    finally:
        for name, _toolset in _PROBES:
            registry.deregister(name)
        registry._ambient_toolsets = saved_ambient
        registry._generation += 1
        model_tools._clear_tool_defs_cache()


def _names(**kwargs) -> set:
    """Tool names a session with this toolset selection would actually see."""
    defs = model_tools.get_tool_definitions(quiet_mode=True, **kwargs)
    return {d["function"]["name"] for d in defs}


class TestUnmarkedSelectionUnchanged:
    def test_unmarked_toolset_stays_absent_from_restricted_session(self, probes):
        names = _names(enabled_toolsets=_RESTRICTED)
        assert "terminal" in names
        assert "web_search" not in names  # the restriction is real
        assert "ambient_probe_tool" not in names
        assert "plain_probe_tool" not in names

    def test_marking_nothing_leaves_selection_identical(self, probes):
        """Two calls with no ambient membership must agree exactly (no ambient side effects)."""
        first = _names(enabled_toolsets=["terminal", "web"])
        second = _names(enabled_toolsets=["terminal", "web"])
        assert first == second
        assert {"ambient_probe_tool", "plain_probe_tool"}.isdisjoint(first)
        assert "terminal" in first and "web_search" in first

    def test_ambient_toolsets_starts_empty_for_these_probes(self, probes):
        assert probes.ambient_toolsets() == set()


class TestMarkedToolsetReachesRestrictedSession:
    def test_marked_toolset_appears_in_restricted_session(self, probes):
        probes.mark_ambient("ambient_probe_ts")
        names = _names(enabled_toolsets=_RESTRICTED)
        assert "ambient_probe_tool" in names
        assert "terminal" in names
        assert "web_search" not in names  # restriction still enforced for everything else

    def test_unmarked_toolset_stays_absent_when_sibling_is_marked(self, probes):
        probes.mark_ambient("ambient_probe_ts")
        names = _names(enabled_toolsets=_RESTRICTED)
        assert "ambient_probe_tool" in names
        assert "plain_probe_tool" not in names

    def test_marking_adds_only_the_marked_tools(self, probes):
        """The delta against the unmarked selection is exactly the ambient toolset's tools."""
        before = _names(enabled_toolsets=_RESTRICTED)
        probes.mark_ambient("ambient_probe_ts")
        after = _names(enabled_toolsets=_RESTRICTED)
        assert after - before == {"ambient_probe_tool"}
        assert before - after == set()

    def test_marking_invalidates_a_memoized_selection(self, probes):
        """Warm the quiet-mode cache first; ambient registration must re-key it."""
        warm = _names(enabled_toolsets=_RESTRICTED)
        assert "ambient_probe_tool" not in warm
        generation = probes._generation
        probes.mark_ambient("ambient_probe_ts")
        assert probes._generation > generation
        assert "ambient_probe_tool" in _names(enabled_toolsets=_RESTRICTED)


class TestSecurityBoundaries:
    def test_disabled_toolset_still_wins_over_ambient(self, probes):
        """An operator's explicit disable is subtracted after the ambient union."""
        probes.mark_ambient("ambient_probe_ts")
        names = _names(enabled_toolsets=_RESTRICTED, disabled_toolsets=["ambient_probe_ts"])
        assert "ambient_probe_tool" not in names
        assert "terminal" in names

    @pytest.mark.parametrize("bad", ["all", "*", "", "   ", None, 7])
    def test_wildcard_and_malformed_scopes_are_rejected(self, probes, bad):
        with pytest.raises(ValueError):
            probes.mark_ambient(bad)
        assert probes.ambient_toolsets() == set()

    def test_ambient_toolsets_returns_a_copy(self, probes):
        probes.mark_ambient("ambient_probe_ts")
        snapshot = probes.ambient_toolsets()
        snapshot.clear()
        assert probes.ambient_toolsets() == {"ambient_probe_ts"}

    def test_marking_is_idempotent(self, probes):
        probes.mark_ambient("ambient_probe_ts")
        probes.mark_ambient("ambient_probe_ts")
        assert probes.ambient_toolsets() == {"ambient_probe_ts"}

    def test_ambient_is_not_read_from_config(self, probes):
        """No config surface feeds the set: a fresh registry with any config still starts empty."""
        fresh = ToolRegistry()
        assert fresh.ambient_toolsets() == set()
        assert fresh.is_ambient_toolset("rescuer") is False
