"""Unit tests for the ported Hindsight graph-expansion primitives.

Pure-function tests: no database, no network, no embedding backend.  Ported
from the upstream ``tests/plugins/memory/test_rekal_hindsight.py`` checks.
"""

import math

from agent.memory.graph import expand_links, link_activation


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_link_activation():
    assert close(link_activation(0), 0.0)
    assert close(link_activation(1), math.tanh(0.5))
    assert link_activation(1) < link_activation(2) < link_activation(3)
    assert link_activation(3) < 1.0
    assert link_activation(1000) <= 1.0
    assert close(link_activation(-5), 0.0)


def test_expand_links():
    adjacency = {
        "a": [("b", "related_to")],
        "b": [("c", "related_to")],
        "c": [("d", "related_to")],
    }
    out = expand_links(["a"], adjacency, max_hops=2)
    assert "a" not in out
    assert "b" in out
    assert "c" in out
    assert "d" not in out
    assert out["b"] > out["c"]
    assert close(out["b"], 0.7)

    strong = expand_links(["a"], {"a": [("b", "supersedes")]}, max_hops=1)
    weak = expand_links(["a"], {"a": [("b", "related_to")]}, max_hops=1)
    assert strong["b"] > weak["b"]
    assert strong["b"] <= 1.0

    assert expand_links([], adjacency) == {}
    assert expand_links(["a"], {}) == {}

    cyclic = {"a": [("b", "related_to")], "b": [("a", "related_to")]}
    assert isinstance(expand_links(["a"], cyclic, max_hops=5), dict)

    wide = {"seed": [(f"n{i}", "related_to") for i in range(100)]}
    assert len(expand_links(["seed"], wide, max_hops=1, budget=10)) <= 10

    assert expand_links(["a"], adjacency, max_hops=5, threshold=0.9) == {}
