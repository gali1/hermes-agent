"""Additive enhanced memory layer for the Hermes agent core.

Pure-algorithm primitives (rank-space fusion, recency decay, temporal query
analysis, graph expansion, deduplication, contradiction detection, evidence
scoring, entity extraction, turn mining, budgeted recall formatting), the
structured SQLite store, and the config-gated backend facade.  Ported from the
OpenCode/MemPalace Hindsight + Rekal implementations and kept stdlib-only.

The layer is opt-in via ``memory.enhanced.enabled`` in config.yaml; when
disabled, :class:`~agent.memory_manager.MemoryManager` behavior is identical to
the built-in + provider path.
"""

from __future__ import annotations

from agent.memory.backend import SYSTEM_PROMPT_BLOCK, EnhancedMemoryBackend
from agent.memory.contradiction import (
    CONTRADICTION_MAX_SIMILARITY,
    CONTRADICTION_MIN_SIMILARITY,
    NEGATION_MARKERS,
    POLARITY_PAIRS,
    detect_contradiction,
)
from agent.memory.context import format_recall_block, select_with_budget
from agent.memory.dedup import (
    is_degenerate,
    is_duplicate,
    normalize_content,
    trigram_set,
    trigram_similarity,
)
from agent.memory.entities import ENTITY_PATTERNS, extract_entities
from agent.memory.evidence import SOURCE_QUALITY, confidence, source_quality
from agent.memory.graph import (
    ENTITY_SATURATION_SCALE,
    RELATION_BOOST,
    SPREAD_DECAY,
    SPREAD_THRESHOLD,
    expand_links,
    link_activation,
)
from agent.memory.mining import (
    PRIVATE_BLOCK_RE,
    SECRET_PATTERNS,
    TRANSIENT_MARKERS,
    classify_candidate,
    is_secret,
    mine_turns,
    strip_private,
)
from agent.memory.ranking import (
    BOOST_LEVELS,
    DEFAULT_RRF_K,
    GRAPH_ALPHA,
    IMPORTANCE_ALPHA,
    LEXICAL_ANCHOR_MARGIN,
    LEXICAL_ANCHOR_MIN_FTS,
    PROOF_ALPHA,
    RECENCY_ALPHA,
    RECENCY_HALFLIFE_DAYS,
    RECENCY_LINEAR_WINDOW_DAYS,
    boosted_rrf_score,
    combined_score,
    compute_recency_decay,
    infer_strategy_boosts,
    proof_norm,
    reciprocal_rank_fusion,
    recency_for_range,
    spans_calendar_period,
)
from agent.memory.schema import (
    MEMORY_SCOPES,
    MEMORY_TYPES,
    SOURCE_TYPES,
    MemoryConflict,
    MemoryEvidence,
    MemoryLink,
    MemoryQuery,
    MemoryRecord,
    MemoryResult,
    MemoryScope,
    MemorySource,
    record_from_row,
)
from agent.memory.store import MemoryStore
from agent.memory.temporal import (
    analyze_query,
    extract_temporal_constraint,
    select_with_temporal_coverage,
    temporal_proximity,
)

__all__ = [
    "BOOST_LEVELS",
    "CONTRADICTION_MAX_SIMILARITY",
    "CONTRADICTION_MIN_SIMILARITY",
    "DEFAULT_RRF_K",
    "LEXICAL_ANCHOR_MIN_FTS",
    "LEXICAL_ANCHOR_MARGIN",
    "ENTITY_PATTERNS",
    "ENTITY_SATURATION_SCALE",
    "EnhancedMemoryBackend",
    "GRAPH_ALPHA",
    "IMPORTANCE_ALPHA",
    "MEMORY_SCOPES",
    "MEMORY_TYPES",
    "MemoryConflict",
    "MemoryEvidence",
    "MemoryLink",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryResult",
    "MemoryScope",
    "MemorySource",
    "MemoryStore",
    "NEGATION_MARKERS",
    "POLARITY_PAIRS",
    "PRIVATE_BLOCK_RE",
    "PROOF_ALPHA",
    "RECENCY_ALPHA",
    "RECENCY_HALFLIFE_DAYS",
    "RECENCY_LINEAR_WINDOW_DAYS",
    "RELATION_BOOST",
    "SECRET_PATTERNS",
    "SOURCE_QUALITY",
    "SOURCE_TYPES",
    "SPREAD_DECAY",
    "SPREAD_THRESHOLD",
    "SYSTEM_PROMPT_BLOCK",
    "TRANSIENT_MARKERS",
    "analyze_query",
    "boosted_rrf_score",
    "infer_strategy_boosts",
    "classify_candidate",
    "combined_score",
    "compute_recency_decay",
    "confidence",
    "detect_contradiction",
    "expand_links",
    "extract_entities",
    "extract_temporal_constraint",
    "format_recall_block",
    "is_degenerate",
    "is_duplicate",
    "is_secret",
    "link_activation",
    "mine_turns",
    "normalize_content",
    "proof_norm",
    "reciprocal_rank_fusion",
    "recency_for_range",
    "record_from_row",
    "select_with_budget",
    "select_with_temporal_coverage",
    "source_quality",
    "spans_calendar_period",
    "strip_private",
    "temporal_proximity",
    "trigram_set",
    "trigram_similarity",
]
