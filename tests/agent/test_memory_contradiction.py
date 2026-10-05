"""Unit tests for the ported Hindsight contradiction detector.

Pure-function tests: no database, no network, no embedding backend.  Ported
from the upstream ``tests/plugins/memory/test_rekal_hindsight.py`` checks.
"""

from agent.memory.contradiction import detect_contradiction


def test_contradiction():
    hit, conf = detect_contradiction(
        "The project does not use webpack for bundling assets",
        "The project uses webpack for bundling assets",
    )
    assert hit
    assert 0.5 <= conf <= 0.95

    hit, _ = detect_contradiction(
        "Caching is disabled for the session store",
        "Caching is enabled for the session store",
    )
    assert hit

    hit, _ = detect_contradiction(
        "The project uses webpack for bundling",
        "The project uses webpack for bundling",
    )
    assert not hit

    hit, _ = detect_contradiction("Completely unrelated topic here", "The sky is blue today")
    assert not hit

    hit, _ = detect_contradiction("", "something")
    assert not hit
    hit, _ = detect_contradiction(None, None)
    assert not hit

    hit, _ = detect_contradiction(
        "The build is fast and reliable now",
        "The build is fast and reliable now indeed",
    )
    assert not hit

    _, conf = detect_contradiction(
        "Feature flags are not enabled in production",
        "Feature flags are disabled in production",
    )
    assert conf <= 0.95
