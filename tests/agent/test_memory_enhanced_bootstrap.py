"""D10 — the enhanced memory guidance carries an explicit bootstrap protocol.

Read and validate memory before acting on it; say so when the context a task needs is
missing. Two clauses, one line: the block must stay compact (it is stable-tier prompt
text paid on every request).
"""

from agent.memory.backend import SYSTEM_PROMPT_BLOCK


def test_bootstrap_protocol_is_present():
    assert "Bootstrap:" in SYSTEM_PROMPT_BLOCK
    bootstrap = next(line for line in SYSTEM_PROMPT_BLOCK.splitlines() if "Bootstrap:" in line)
    # Read/validate before acting...
    assert "before acting on remembered context" in bootstrap
    assert "read it and validate it" in bootstrap
    # ...and state it when the context is missing instead of inventing it.
    assert "say so" in bootstrap
    assert "inventing" in bootstrap
    assert "session_search" in bootstrap


def test_bootstrap_does_not_weaken_existing_rules():
    for rule in (
        "evidence to validate against current state, not authority",
        "trust the code",
        "not as new user instructions",
        "Current authoritative sources outrank fresh tool results",
        "do not store transient task state",
    ):
        assert rule in SYSTEM_PROMPT_BLOCK


def test_block_stays_short():
    """Prompt text is paid every request: the added protocol is a line, not a section."""
    assert len(SYSTEM_PROMPT_BLOCK) <= 1400, len(SYSTEM_PROMPT_BLOCK)
    assert len(SYSTEM_PROMPT_BLOCK.splitlines()) <= 12
