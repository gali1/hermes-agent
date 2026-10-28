"""Lexical contradiction detection between candidate memories.

Started life as a verbatim port of the OpenCode/MemPalace Hindsight primitives
(upstream ``plugins/memory/rekal/mempalace_hindsight.py``) and has since been
localised: the original compared *whole documents*, which made almost any two
long messages look like a disagreement.  Stdlib only and pure.

The defect that motivated the rewrite: negation was measured across the entire
text, so a 1,000-character status report that happens to contain "can't" once
was compared against a sibling report that did not, and the pair was recorded as
a contradiction even though the only real difference was a timestamp.  A
contradiction needs the negation to sit *on the claim that differs*, so the
comparison now happens on the single most similar sentence pair, and only the
tokens that actually differ are allowed to vote.
"""

from __future__ import annotations

import re
from collections import Counter

from agent.memory.dedup import _TRGM_WORD, trigram_similarity

# Surface markers of negation/reversal. Lexical detection cannot prove a
# contradiction, so callers treat a hit as a candidate for review (a
# `contradicts` link) rather than as grounds to delete or rewrite anything.
NEGATION_MARKERS = (
    "not ", "no longer ", "doesn't ", "does not ", "isn't ", "is not ",
    "aren't ", "are not ", "won't ", "will not ", "never ", "cannot ", "can't ",
    "removed ", "stopped ", "dropped ", "disabled ", "deprecated ", "reverted ",
    "instead of ", "rather than ", "switched from ", "moved away from ",
)

# The same markers as bare words, for token-level comparison.
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "cannot", "can't", "cant", "won't", "wont",
    "isn't", "isnt", "aren't", "arent", "doesn't", "doesnt", "didn't", "didnt",
    "don't", "dont", "none", "nothing", "removed", "removed", "stopped",
    "dropped", "disabled", "deprecated", "reverted", "without", "nor",
})

# Antonym pairs that flip a statement's meaning without any negation word.
POLARITY_PAIRS = (
    ("enable", "disable"), ("enabled", "disabled"), ("allow", "deny"),
    ("allowed", "denied"), ("add", "remove"), ("added", "removed"),
    ("include", "exclude"), ("start", "stop"), ("started", "stopped"),
    ("on", "off"), ("true", "false"), ("always", "never"),
    ("support", "unsupported"), ("prefer", "avoid"),
)

_POLARITY_INDEX = {}
for _left, _right in POLARITY_PAIRS:
    _POLARITY_INDEX.setdefault(_left, set()).add(_right)
    _POLARITY_INDEX.setdefault(_right, set()).add(_left)

CONTRADICTION_MIN_SIMILARITY = 0.35
CONTRADICTION_MAX_SIMILARITY = 0.97

# Candidates are queued for adjudication rather than linked, so the bar is a
# shared *subject* rather than shared surface: paraphrase is defined by lower
# word overlap, so a similarity floor would exclude exactly the class this is
# for. Two shared content tokens is the cheapest faithful proxy for "same
# subject".
CONTRADICTION_CANDIDATE_MIN_SHARED_TOKENS = 2

# Contradiction is a property of atomic propositions.  Whole-turn captures (a
# 56 KB prompt dump, a skill inventory, a build log) share vocabulary and
# negation words with unrelated notes, which is how long documents used to
# produce `contradicts` links that were pure noise.  Sentence localisation
# already fixes most of that; this stays as a cheap outer guard.
CONTRADICTION_MAX_CHARS = 1200

# Sentence boundaries.  Split on terminators and newlines, not on abbreviations
# in a smarter way -- over-splitting only ever costs recall on contrived text,
# while under-splitting costs precision on real ones.
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

# `_TRGM_WORD` splits on non-word characters, so "isn't" reaches the token level
# as "isn" + "t" and stops matching the negation lexicon entirely. Expanding
# contractions before tokenising is what makes the negation rule work on the way
# people actually write.
_CANT_RE = re.compile(r"\bcan't\b", re.IGNORECASE)
_NOT_CONTRACTION_RE = re.compile(
    r"\b(isn|aren|wasn|weren|doesn|didn|don|won|couldn|shouldn|wouldn|hasn|haven|ain)['\u2019]t\b",
    re.IGNORECASE,
)
# The stem before "'t" is not a word on its own, so each one expands to the
# auxiliary it actually means.
_CONTRACTION_STEMS = {
    "isn": "is", "aren": "are", "wasn": "was", "weren": "were",
    "doesn": "does", "didn": "did", "don": "do", "won": "will",
    "couldn": "could", "shouldn": "should", "wouldn": "would",
    "hasn": "has", "haven": "have", "ain": "is",
}


def _expand_contractions(text):
    expanded = _CANT_RE.sub("can not", text or "")
    return _NOT_CONTRACTION_RE.sub(
        lambda match: f"{_CONTRACTION_STEMS.get(match.group(1).lower(), match.group(1))} not",
        expanded,
    )

# Two sentences are the "same claim" when their residue (everything except the
# polarity difference) still looks like the same sentence.  Below this they are
# talking about different things and must never be paired.
CONTRADICTION_RESIDUE_SIMILARITY = 0.55


def _has_negation(text):
    padded = f" {text.lower()} "
    return any(marker in padded for marker in NEGATION_MARKERS)


def _polarity_conflict(a_low, b_low):
    a_words = set(_TRGM_WORD.findall(a_low))
    b_words = set(_TRGM_WORD.findall(b_low))
    for left, right in POLARITY_PAIRS:
        if (left in a_words and right in b_words) or (right in a_words and left in b_words):
            return True
    return False


def _sentences(text):
    """Split into candidate claims, trimmed and non-empty."""
    return [part.strip() for part in _SENTENCE_RE.split(text or "") if part.strip()]


def _tokens(text):
    return _TRGM_WORD.findall((text or "").lower())


# Auxiliaries and copulas carry no claim of their own: "uses" and "does not use"
# differ by a negation, not by the verb, and "is enabled" / "isn't enabled"
# differ by the contraction alone. Normalising them before the difference is
# computed is what stops the subset rule from rejecting real contradictions on
# inflection.
_AUXILIARIES = frozenset({
    "do", "does", "did", "has", "have", "had",
    "be", "been", "being", "is", "are", "was", "were", "am",
})

# State predicates collapse to one canonical token per polarity. This is what
# lets "caching is off" match "caching is disabled" without a model: adjectives
# that mean off/on are mapped, then the rest of the sentence is stemmed, so
# inflection and word choice stop hiding the same claim.
_STATE_NEG = frozenset({
    "off", "disabled", "disable", "inactive", "closed", "absent", "missing",
    "unavailable", "unsupported", "none", "empty", "unset", "broken", "dead",
})
_STATE_POS = frozenset({
    "on", "enabled", "enable", "active", "open", "present", "available",
    "supported", "set", "configured", "working", "alive", "up",
})

NEG_STATE_TOKEN = "__neg_state__"
POS_STATE_TOKEN = "__pos_state__"


def _stem(token):
    """Fold common English inflection so caching/cache/caches compare equal."""
    if len(token) <= 3:
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def _canonical_tokens(text):
    """Comparison tokens: state words collapsed, inflections folded, aux dropped."""
    canonical = []
    for raw in _tokens(text):
        if raw in _STATE_NEG:
            canonical.append(NEG_STATE_TOKEN)
            continue
        if raw in _STATE_POS:
            canonical.append(POS_STATE_TOKEN)
            continue
        key = _normalise_token(raw)
        if not key:
            continue
        canonical.append(_stem(key))
    return canonical


def _is_state_token(token):
    return token in (NEG_STATE_TOKEN, POS_STATE_TOKEN)


def _state_polarity_conflict(a_tokens, b_tokens):
    return (NEG_STATE_TOKEN in a_tokens and POS_STATE_TOKEN in b_tokens) or (
        POS_STATE_TOKEN in a_tokens and NEG_STATE_TOKEN in b_tokens
    )


# Intensifiers and hedges that cannot change what is being claimed. Permitting
# them in the difference recovers contradictions that add a word ("not enabled
# at all") without letting a noun, number or filename sneak through.
_NON_CLAIM_TOKENS = frozenset({
    "at", "all", "any", "more", "yet", "still", "actually", "currently", "now",
    "anymore", "quite", "very", "only", "just", "also", "even", "again",
    "already", "ever", "really", "truly", "exactly", "simply", "sure", "same",
})


def _normalise_token(token):
    """Comparison key for a token: auxiliaries vanish, simple plurals fold."""
    if token in _AUXILIARIES:
        return ""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _compare_tokens(text):
    keys = (_normalise_token(token) for token in _tokens(text))
    return [key for key in keys if key]


def _best_sentence_pair(a_text, b_text):
    """Return the most similar (similarity, a_sentence, b_sentence).

    Contradiction is a claim-level property, so the pair of sentences that share
    the most vocabulary is the only place a disagreement can live.
    """
    a_sentences = _sentences(a_text) or [a_text]
    b_sentences = _sentences(b_text) or [b_text]
    best = (0.0, a_sentences[0], b_sentences[0])
    for a_sentence in a_sentences:
        for b_sentence in b_sentences:
            similarity = trigram_similarity(a_sentence, b_sentence)
            if similarity > best[0]:
                best = (similarity, a_sentence, b_sentence)
    return best


def _differing_tokens(a_sentence, b_sentence):
    """Tokens present on one side but not the other, as a multiset difference."""
    a_tokens = Counter(_canonical_tokens(a_sentence))
    b_tokens = Counter(_canonical_tokens(b_sentence))
    differing = set()
    differing |= set((a_tokens - b_tokens))
    differing |= set((b_tokens - a_tokens))
    return differing


def _residue(sentence, drop):
    return " ".join(key for key in _canonical_tokens(sentence) if key not in drop)


def classify_pair(new_content, existing_content,
                  min_similarity=CONTRADICTION_MIN_SIMILARITY,
                  max_similarity=CONTRADICTION_MAX_SIMILARITY):
    """Three-way verdict: 'contradiction', 'candidate' or 'unrelated'.

    `candidate` is the important one. Lexical rules can prove a flipped bit;
    they cannot prove that "caching is off" and "the store caches every request"
    disagree, because the two sentences share no structure. Rather than
    guess (and flood the review queue with noise) or drop it (and miss the
    contradiction), an unprovable-but-suspicious pair is recorded as a candidate
    for adjudication. That is the division of labour: rules decide what they can
    prove, the model decides the rest, and nothing is silently ignored.
    """
    if not new_content or not existing_content:
        return "unrelated", 0.0

    new_content = _expand_contractions(new_content)
    existing_content = _expand_contractions(existing_content)

    if len(new_content) > CONTRADICTION_MAX_CHARS or len(existing_content) > CONTRADICTION_MAX_CHARS:
        return "unrelated", 0.0

    similarity, a_sentence, b_sentence = _best_sentence_pair(new_content, existing_content)
    if similarity < min_similarity or similarity > max_similarity:
        return "unrelated", 0.0

    a_low = a_sentence.lower()
    b_low = b_sentence.lower()
    a_canonical = _canonical_tokens(a_sentence)
    b_canonical = _canonical_tokens(b_sentence)

    drop = set(_NEGATION_TOKENS)
    for left, right in POLARITY_PAIRS:
        drop.add(left)
        drop.add(right)
    drop.add(NEG_STATE_TOKEN)
    drop.add(POS_STATE_TOKEN)
    allowed = drop | _NON_CLAIM_TOKENS

    differing = _differing_tokens(a_sentence, b_sentence)
    polarity_in_diff = _state_polarity_conflict(a_canonical, b_canonical) or any(
        token in _NEGATION_TOKENS or _is_state_token(token) for token in differing
    )

    if differing and differing.issubset(allowed) and polarity_in_diff:
        negation_differs = any(token in _NEGATION_TOKENS for token in differing)
        polarity_differs = _polarity_conflict(a_low, b_low) or _state_polarity_conflict(
            a_canonical, b_canonical
        )
        residue_a = _residue(a_sentence, allowed)
        residue_b = _residue(b_sentence, allowed)
        if residue_a and residue_b and trigram_similarity(
            residue_a, residue_b
        ) >= CONTRADICTION_RESIDUE_SIMILARITY:
            confidence = 0.5 + 0.3 * similarity
            if negation_differs and polarity_differs:
                confidence += 0.15
            return "contradiction", round(min(0.95, confidence), 3)

    # Unprovable, but is there a polarity hint at all? One side carrying a
    # negation or a negative state while the other carries neither is exactly
    # the shape of a paraphrase disagreement.
    #
    # Every candidate costs a review, so require a shared subject: at least two
    # content tokens in common (polarity/filler tokens do not count). A
    # similarity floor would be wrong here -- paraphrase is exactly the case
    # with low surface overlap.
    a_has_neg = any(tok in _NEGATION_TOKENS for tok in a_canonical) or NEG_STATE_TOKEN in a_canonical
    b_has_neg = any(tok in _NEGATION_TOKENS for tok in b_canonical) or NEG_STATE_TOKEN in b_canonical
    if a_has_neg != b_has_neg:
        shared_content = (set(a_canonical) & set(b_canonical)) - set(allowed)
        if len(shared_content) >= CONTRADICTION_CANDIDATE_MIN_SHARED_TOKENS:
            return "candidate", round(0.3 + 0.2 * similarity, 3)
    return "unrelated", 0.0


def detect_contradiction(new_content, existing_content,
                         min_similarity=CONTRADICTION_MIN_SIMILARITY,
                         max_similarity=CONTRADICTION_MAX_SIMILARITY):
    """Backwards-compatible wrapper: True only for a provable contradiction."""
    verdict, confidence = classify_pair(
        new_content, existing_content,
        min_similarity=min_similarity, max_similarity=max_similarity,
    )
    return verdict == "contradiction", confidence
