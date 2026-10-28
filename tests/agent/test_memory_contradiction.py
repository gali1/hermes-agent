"""Unit tests for the ported Hindsight contradiction detector.

Pure-function tests: no database, no network, no embedding backend.  Ported
from the upstream ``tests/plugins/memory/test_rekal_hindsight.py`` checks.
"""

from agent.memory.contradiction import CONTRADICTION_MAX_CHARS, detect_contradiction


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


def test_long_documents_are_never_paired():
    """A prompt dump or inventory shares vocabulary with unrelated notes.

    Pairing multi-KB documents on trigram similarity plus a negation word is how
    the store accumulated `contradicts` links that were pure noise, so anything
    above the atomic-statement threshold is refused outright.
    """
    shared = "gateway session transport websocket layer design pattern "
    long_note = (shared * 60) + "this is not the real fix"
    assert len(long_note) > CONTRADICTION_MAX_CHARS
    hit, conf = detect_contradiction(long_note, shared * 60)
    assert not hit
    assert conf == 0.0


def test_short_statements_still_pair():
    """The length gate must not blunt detection at atomic sizes."""
    hit, _ = detect_contradiction(
        "Caching is disabled for the session store",
        "Caching is enabled for the session store",
    )
    assert hit


def test_revision_with_changed_details_is_not_a_contradiction():
    """Two status reports differing only in a timestamp are a revision.

    This is the live false positive that motivated sentence localisation: the
    original detector measured negation across the whole document, so a "can't"
    in an unrelated sentence plus a shared vocabulary was enough to pair them.
    """
    earlier = (
        "Re-armed. Fires 19:04 - now 19:02:42, so ~75 seconds. "
        "Entry: 04 19 27 10 * script (line 57, pinned to today). "
        "Backup: crontab.bak.20261027-190242. "
        "The script removes its own line first so a failed restart can't loop. "
        "Expect this chat to go quiet for about ten seconds while the unit cycles."
    )
    later = (
        "Armed. Fires 20:45 - now 20:43:48, so ~70 seconds. "
        "Entry: 45 20 27 10 * script (line 59, pinned to today). "
        "Backup: crontab.bak.20261027-204348. "
        "The script removes its own line first so a failed restart cannot loop. "
        "Expect this chat to go quiet for about ten seconds while the unit cycles."
    )
    hit, conf = detect_contradiction(later, earlier)
    assert not hit, conf


def test_negation_must_sit_on_the_differing_claim():
    """A negation elsewhere in the text is not evidence about this claim."""
    a = (
        "The routing hotfix is committed at fac3b310c and the tree is clean. "
        "The push is still blocked, so I can't finish it from here. "
        "Both commits are ready at the tip of master."
    )
    b = (
        "The routing hotfix is committed at 996ed2a36 and the tree is clean. "
        "The push is still blocked, so I can't finish it from here. "
        "Both commits are ready at the tip of master."
    )
    hit, conf = detect_contradiction(a, b)
    assert not hit, conf


def test_negation_inside_the_differing_claim_is_still_caught():
    """Localisation must not cost the real signal."""
    hit, _ = detect_contradiction(
        "The gateway restarts cleanly after the drain completes",
        "The gateway does not restart cleanly after the drain completes",
    )
    assert hit

    hit, _ = detect_contradiction(
        "Session caching is enabled for the store",
        "Session caching is disabled for the store",
    )
    assert hit


def test_intensifiers_do_not_hide_a_contradiction():
    """Recall recovery: added hedge words must not defeat detection."""
    hit, conf = detect_contradiction(
        "Caching is not enabled at all for the session store",
        "Caching is enabled for the session store",
    )
    assert hit, conf

    hit, conf = detect_contradiction(
        "The gateway isn't enabled in production",
        "The gateway is enabled in production",
    )
    assert hit, conf


def test_a_changed_noun_is_still_not_a_contradiction():
    """Precision guard: a filler allowance must not become a noun allowance."""
    hit, conf = detect_contradiction(
        "Caching is not enabled for the session store",
        "Caching is not enabled for the analytics pipeline",
    )
    assert not hit, conf
