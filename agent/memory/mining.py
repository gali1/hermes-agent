"""Turn mining: secret rejection, transient filtering, and classification.

Ported from the OpenCode/MemPalace Hindsight primitives.  Stdlib only and
side-effect free: strips private blocks and injected context tags, rejects
secret-bearing and transient text, and classifies the remainder into a memory
type.
"""

from __future__ import annotations

import re

from agent.memory.dedup import is_degenerate, normalize_content

TRANSIENT_MARKERS = frozenset({
    "hi", "hello", "hey", "yo", "thanks", "thank you", "thx", "ty", "cheers",
    "ok", "okay", "k", "sure", "got it", "sounds good", "nice", "cool", "great",
    "awesome", "perfect", "no problem", "you're welcome", "good morning",
    "good afternoon", "good evening", "good night", "bye", "goodbye",
    "see you", "see ya", "lol", "haha", "welcome", "yes", "no", "yep", "nope",
    "yeah", "alright",
})

_TRANSIENT_WORDS = frozenset({
    "a", "afternoon", "all", "alright", "am", "an", "and", "anytime", "appreciate",
    "are", "awesome", "be", "been", "being", "bye", "cheers", "cool", "done",
    "evening", "for", "good", "goodbye", "got", "great", "haha", "hello", "help",
    "hey", "hi", "hmm", "i", "is", "it", "k", "later", "lol", "me", "morning",
    "much", "nice", "night", "no", "nope", "ok", "okay", "perfect", "please",
    "problem", "really", "right", "see", "so", "sounds", "sure", "thank",
    "thanks", "that", "thats", "that's", "the", "thx", "today", "ty", "um",
    "very", "was", "we", "welcome", "well", "were", "wow", "ya", "yeah", "yep",
    "yes", "you", "your", "you're", "yup",
})

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*\S{8,}"),
]

PRIVATE_BLOCK_RE = re.compile(r"<private>.*?</private>", re.IGNORECASE | re.DOTALL)

_CONTEXT_TAGS_RE = re.compile(
    r"<mempalace_context>.*?</mempalace_context>"
    r"|<rekal-context>.*?</rekal-context>"
    r"|<memory-context>.*?</memory-context>",
    re.IGNORECASE | re.DOTALL,
)

_CATEGORY_PATTERNS = (
    ("preference", re.compile(
        r"\b(prefer\w*|like|likes|liked|always|never|favorite|favourite)\b",
        re.IGNORECASE,
    )),
    ("decision", re.compile(
        r"\b(decid\w*|cho[so]\w*|will use|going with|opted|selected|select)\b",
        re.IGNORECASE,
    )),
    ("procedure", re.compile(
        r"\b(step|steps|first|then|process|workflow|procedures?)\b",
        re.IGNORECASE,
    )),
    ("bug", re.compile(
        r"\b(error\w*|bugs?|crash\w*|fail\w*|regressions?)\b",
        re.IGNORECASE,
    )),
    ("solution", re.compile(
        r"\b(fix\w*|solv\w*|resolv\w*|workaround\w*|patch\w*)\b",
        re.IGNORECASE,
    )),
    ("architecture", re.compile(
        r"\b(architecture|architectural|designs?|conventions?|patterns?)\b",
        re.IGNORECASE,
    )),
    ("configuration", re.compile(
        r"\b(config|configs|configuration|settings?|flags?|env)\b",
        re.IGNORECASE,
    )),
    ("relationship", re.compile(
        r"\b(depends on|owns|uses|connect\w*)\b",
        re.IGNORECASE,
    )),
)


def is_secret(text) -> bool:
    """True when `text` carries an API key, token, or credential assignment."""
    if not isinstance(text, str) or not text:
        return False
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def strip_private(text) -> str:
    """Remove `<private>...</private>` blocks, case-insensitively."""
    if not isinstance(text, str) or not text:
        return ""
    return PRIVATE_BLOCK_RE.sub(" ", text)


def _is_transient(text) -> bool:
    normalized = normalize_content(text)
    if not normalized:
        return True
    core = normalized.strip("!?.,;:'\" ")
    if core in TRANSIENT_MARKERS:
        return True
    words = re.findall(r"[a-z0-9']+", core)
    return bool(words) and len(core) < 60 and all(word in _TRANSIENT_WORDS for word in words)


def classify_candidate(text) -> str | None:
    """Classify candidate text into a memory type, or None when not worth storing."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if len(stripped) < 20:
        return None
    if is_degenerate(stripped):
        return None
    if is_secret(stripped):
        return None
    if _is_transient(stripped):
        return None
    for memory_type, pattern in _CATEGORY_PATTERNS:
        if pattern.search(stripped):
            return memory_type
    return "fact"


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def mine_turns(messages, *, min_length=20, limit=20) -> list[dict]:
    """Mine OpenAI-style message dicts into deduplicated memory candidates."""
    if not messages or limit <= 0:
        return []

    mined = []
    seen = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_text(message.get("content"))
        if not text:
            continue
        text = strip_private(text)
        text = _CONTEXT_TAGS_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or len(text) < min_length:
            continue
        if is_secret(text):
            continue
        memory_type = classify_candidate(text)
        if memory_type is None:
            continue
        normalized = normalize_content(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        mined.append({"content": text, "memory_type": memory_type, "role": role})
        if len(mined) >= limit:
            break
    return mined
