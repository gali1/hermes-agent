"""B10 — enhanced actions are advertised only when the enhanced layer is enabled.

All seven enhanced actions used to sit in the ``memory`` tool enum unconditionally, so a
disabled layer advertised capability that could only answer "Enhanced memory is not
enabled". The surface now follows ``memory.enhanced.enabled`` (on by default), including
the parameters only those actions read.
"""

import pytest
import hermes_yaml as yaml

from model_tools import get_tool_definitions

ALL_ENHANCED = {"search", "recall", "conflicts", "timeline", "topics", "health", "reinforce"}
ENHANCED_ONLY_PARAMS = {"query", "limit", "memory_type", "graph_expand", "temporal", "memory_id"}


@pytest.fixture(autouse=True)
def _clear_caches():
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_memory_config(home, **memory_section):
    (home / "config.yaml").write_text(
        yaml.safe_dump({"memory": memory_section}), encoding="utf-8")


def _memory_definition():
    return next(
        tool["function"]
        for tool in get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)
        if tool["function"]["name"] == "memory"
    )


def test_enhanced_actions_hidden_when_layer_disabled(hermes_home):
    _write_memory_config(hermes_home, enhanced={"enabled": False})
    definition = _memory_definition()
    enum = definition["parameters"]["properties"]["action"]["enum"]

    assert ALL_ENHANCED.isdisjoint(enum)
    assert enum == ["add", "replace", "remove"]
    assert "ENHANCED" not in definition["description"]
    assert "Enhanced memory is not enabled" not in definition["description"]


def test_enhanced_only_params_hidden_with_the_actions(hermes_home):
    _write_memory_config(hermes_home, enhanced={"enabled": False})
    properties = set(_memory_definition()["parameters"]["properties"])

    assert ENHANCED_ONLY_PARAMS.isdisjoint(properties)
    assert {"action", "target", "content", "old_text", "new_text", "operations"} <= properties


def test_enhanced_actions_advertised_when_layer_enabled(hermes_home):
    _write_memory_config(hermes_home, enhanced={"enabled": True})
    definition = _memory_definition()
    enum = set(definition["parameters"]["properties"]["action"]["enum"])

    assert ALL_ENHANCED <= enum
    assert {"add", "replace", "remove"} <= enum
    assert ENHANCED_ONLY_PARAMS <= set(definition["parameters"]["properties"])
    assert "ENHANCED (when memory.enhanced is enabled)" in definition["description"]


def test_enhanced_actions_advertised_by_default(hermes_home):
    """The shipped default is on, so an untouched config keeps the full surface."""
    enum = set(_memory_definition()["parameters"]["properties"]["action"]["enum"])
    assert ALL_ENHANCED <= enum
