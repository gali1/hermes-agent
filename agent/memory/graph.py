"""Graph / link expansion via spreading activation.

Ported verbatim from the OpenCode/MemPalace Hindsight primitives (upstream
``plugins/memory/rekal/mempalace_hindsight.py``).  Stdlib only and pure.
"""

from __future__ import annotations

import math

# Saturating so the first shared connection carries most of the signal:
#   1 link -> 0.46   2 -> 0.76   3 -> 0.91   4 -> 0.96
ENTITY_SATURATION_SCALE = 0.5

# Per-hop decay for spreading activation, and the floor below which a node stops
# propagating. Both bound BFS cost as much as they shape relevance.
SPREAD_DECAY = 0.7
SPREAD_THRESHOLD = 0.2

# Relation multipliers: an explicit supersedes/contradicts edge is a much
# stronger statement about relatedness than a generic association.
RELATION_BOOST = {
    "supersedes": 2.0,
    "contradicts": 1.5,
    "related_to": 1.0,
}


def link_activation(shared_count, scale=ENTITY_SATURATION_SCALE):
    """Shared-connection count -> saturating activation in [0,1)."""
    return math.tanh(max(0, shared_count) * scale)


def expand_links(seed_ids, adjacency, max_hops=2, budget=50,
                 decay=SPREAD_DECAY, threshold=SPREAD_THRESHOLD):
    """Spreading activation over the memory-link graph.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    `adjacency` maps memory id -> iterable of (neighbour_id, relation).  Returns
    {memory_id: activation} for newly reached nodes only; seeds are excluded
    because they already entered retrieval through another arm and re-scoring
    them here would double-count.

    Activation propagates as `parent * relation_boost * decay`, and a node stops
    expanding once it drops below `threshold`.  Both `budget` and `max_hops`
    bound the traversal independently so a densely linked store cannot stall a
    search.
    """
    if not seed_ids or not adjacency:
        return {}

    visited = set(seed_ids)
    activations = {}
    frontier = [(sid, 1.0) for sid in seed_ids]

    for _ in range(max(1, max_hops)):
        if not frontier or len(activations) >= budget:
            break
        next_frontier = []
        for node_id, parent_activation in frontier:
            for neighbour_id, relation in adjacency.get(node_id, ()):
                if neighbour_id in visited:
                    continue
                boost = RELATION_BOOST.get(relation, 1.0)
                propagated = parent_activation * boost * decay
                if propagated <= threshold:
                    continue
                # A node reachable by several paths keeps its strongest.
                if propagated > activations.get(neighbour_id, 0.0):
                    activations[neighbour_id] = min(1.0, propagated)
                if len(activations) >= budget:
                    break
                next_frontier.append((neighbour_id, propagated))
            if len(activations) >= budget:
                break
        visited.update(node_id for node_id, _ in next_frontier)
        frontier = next_frontier

    return activations
