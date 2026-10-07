"""Net-cost gate for transcript mutation (Headroom-derived, Phase 3c).

Ports Headroom's cache-aware mutation formula (#856). Removing
``delta_tokens`` from a live transcript forces a cache write of the new
suffix, so the mutation only pays off when the expected savings over the
remaining reads cover that penalty:

    gain = delta * (w + r*(R - 1)) - p_alive * (w - r) * (S + delta)

where ``w`` is the cache-write multiplier, ``r`` the cache-read multiplier,
``R`` the expected remaining reads, ``p_alive`` the probability the cache is
still alive, and ``S`` the summary (suffix) token count.

Opt-in: the context compressor consults :func:`should_compress` only when
``compression.content_aware.net_cost_gate`` is true (mirroring Headroom's
``HEADROOM_NET_COST_POLICY``). Pure and deterministic.
"""

from __future__ import annotations

import math

__all__ = [
    "DEFAULT_EXPECTED_READS",
    "DEFAULT_P_ALIVE",
    "DEFAULT_WRITE_COST",
    "DEFAULT_READ_COST",
    "net_mutation_gain",
    "should_compress",
]

DEFAULT_EXPECTED_READS = 10.0
DEFAULT_P_ALIVE = 1.0
DEFAULT_WRITE_COST = 1.25
DEFAULT_READ_COST = 0.1


def _non_negative(value) -> float:
    number = float(value)
    if math.isnan(number):
        return 0.0
    return max(0.0, number)


def _probability(value) -> float:
    number = float(value)
    if math.isnan(number):
        return 1.0
    return min(max(number, 0.0), 1.0)


def net_mutation_gain(
    delta_tokens,
    summary_tokens,
    expected_reads=DEFAULT_EXPECTED_READS,
    p_alive=DEFAULT_P_ALIVE,
    write_cost=DEFAULT_WRITE_COST,
    read_cost=DEFAULT_READ_COST,
) -> float:
    """Net gain (in plain-input-token cost units) of a transcript mutation.

    ``delta_tokens`` is how many tokens the mutation removes and
    ``summary_tokens`` the suffix it writes in their place. Inputs are
    clamped: delta/summary/reads to ``>= 0`` (NaN to 0), ``p_alive`` to
    ``[0, 1]`` (NaN to 1, the conservative full-penalty assumption).
    """
    delta = _non_negative(delta_tokens)
    suffix = _non_negative(summary_tokens)
    reads = _non_negative(expected_reads)
    alive = _probability(p_alive)
    write = float(write_cost)
    read = float(read_cost)
    return delta * (write + read * (reads - 1.0)) - alive * (write - read) * (suffix + delta)


def should_compress(delta_tokens, summary_tokens, **kwargs) -> bool:
    """True when the mutation's net gain is strictly positive."""
    return net_mutation_gain(delta_tokens, summary_tokens, **kwargs) > 0.0
