"""Evidence quality and bounded confidence scoring.

Ported from the OpenCode/MemPalace Hindsight primitives.  Stdlib only and
deterministic: confidence combines proof strength, source quality, temporal
consistency, and an explicit contradiction penalty into [0.0, 0.95].
"""

from __future__ import annotations

from agent.memory.ranking import proof_norm

SOURCE_QUALITY = {
    "user": 0.9,
    "conversation": 0.7,
    "tool": 0.6,
    "file": 0.8,
    "git": 0.8,
    "agent": 0.5,
    "memory": 0.6,
    "inferred": 0.3,
    "graph": 0.3,
}

_DEFAULT_SOURCE_QUALITY = 0.5


def source_quality(source_type) -> float:
    """Map a source type to its quality signal, defaulting to 0.5."""
    if not source_type:
        return _DEFAULT_SOURCE_QUALITY
    return SOURCE_QUALITY.get(str(source_type).lower(), _DEFAULT_SOURCE_QUALITY)


def confidence(proof_count=1, source_quality_value=0.5, contradiction_penalty=0.0,
               temporal_consistency=0.5) -> float:
    """Bounded [0.0, 0.95] confidence from evidence signals, each neutral at 0.5."""
    value = (
        0.5
        + 0.25 * (proof_norm(proof_count) - 0.5)
        + 0.2 * (source_quality_value - 0.5)
        + 0.1 * (temporal_consistency - 0.5)
        - contradiction_penalty
    )
    return min(0.95, max(0.0, value))
