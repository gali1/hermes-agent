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
    # OpenAI-style and prefixed keys.  `sk-` alone only matched bare alnum
    # tokens, so `sk-proj-…`, `sk-ant-…` and `sk_live_…` slipped through; the
    # character class now covers the separators those families use.
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk_live_[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)hf_[A-Za-z0-9]{30,}"),
    re.compile(r"(?i)nvapi-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"(?i)(?:pcsk|whsec|shpat|dop_v1|rnd)_[A-Za-z0-9_\-]{16,}"),
    # OpenCode / provider-scoped keys: `oc_sk_<id>_<id>_<secret>`.
    re.compile(r"oc_sk_[A-Za-z0-9_\-]{16,}"),
    # Bare JWT (three base64url segments) — access tokens are routinely pasted.
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*\S{8,}"),
    # Prose form: "update the opencode key with 'oc_…'" / "the token is …".
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password|credential)s?\b"
        r"[^\n]{0,32}?\b(?:is|was|with|to|as|=|:)\b[^\n]{0,8}?['\"]?[A-Za-z0-9_\-]{20,}"
    ),
]

# Auto-capture sees whole chat turns, including injected system/agent prompts and
# pasted transcripts.  Those are instruction text, not durable facts about the
# user's world, and storing them is what used to flood recall.
_INJECTED_MARKERS = (
    "[important:",
    "[system note",
    "[out-of-band user message",
    "[/out-of-band user message]",
    "[cron_failure]",
    "recalled memory (background evidence",
    "you are running as a scheduled cron job",
    "agent instructions were loaded from",
    "consolidated prompt",
    "do not use send_message",
    "reply with only the word",
)

_INJECTED_RES = (
    # Prompt-shaped openers: "You are <role>, …" for a system/agent preamble.
    re.compile(r"(?i)^\s*(?:you are|act as)\b.{0,240}?\b(?:assistant|agent|chief of staff|orchestrator)\b"),
    # Leftover injected-context fences from a broken/truncated block.
    re.compile(r"(?i)</?(?:memory-context|mempalace_context|rekal-context)\b"),
)

# An atomic memory is short. Whole-turn captures (prompt dumps, build logs, full
# skill inventories) are not, and multi-KB blobs are what made contradiction
# detection produce noise.  Set above the longest genuinely useful note seen
# (a 2.8 KB config write-up) so it clips dumps without clipping real notes.
MAX_CANDIDATE_CHARS = 3000

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


def looks_injected(text) -> bool:
    """True when `text` is injected instruction text rather than user knowledge.

    Auto-capture receives whole turns, so system preambles, cron wrappers and
    agent role prompts reach the miner looking like ordinary prose.  Storing
    them pollutes recall with instructions that were never facts.
    """
    if not isinstance(text, str) or not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _INJECTED_MARKERS):
        return True
    return any(pattern.search(text) for pattern in _INJECTED_RES)


def _is_transient(text) -> bool:
    normalized = normalize_content(text)
    if not normalized:
        return True
    core = normalized.strip("!?.,;:'\" ")
    if core in TRANSIENT_MARKERS:
        return True
    words = re.findall(r"[a-z0-9']+", core)
    return bool(words) and len(core) < 60 and all(word in _TRANSIENT_WORDS for word in words)


# -- Requests and status lines: the classes heuristics used to miss -----------
# A turn is either *knowledge* ("the gateway owns the websocket transport") or a
# *move in the conversation* ("all features working?").  Only the first belongs
# in a memory, but mining sees both and stored the second, which then matched
# every later query through FTS prefix expansion.  Both filters below are pure
# regex and run in microseconds — no model call, no tokens.
MAX_REQUEST_CHARS = 200
# A one-line question is a request.  A long paragraph that happens to end in
# "?" is a note with a question appended, so length alone (not the "?") decides
# unless the paragraph opens by interrogating.
MAX_SHORT_QUESTION_CHARS = 120
# Without a "?" an interrogative opener is much weaker evidence — "When the user
# selects vtcode_wsl … it only shows one model" is a bug report, not a question.
MAX_UNMARKED_QUESTION_CHARS = 80

# Interrogative openers only count as a question when the apostrophe is absent:
# "Can't reproduce…" leads with "can" but is a claim, not a question.
_LEADING_INTERROGATIVE = re.compile(
    r"(?i)^\s*(?:what|which|who|whom|whose|where|when|why|how|whether|"
    r"(?:is|are|was|were|am|do|does|did|can|could|should|would|will|shall|"
    r"have|has|had)\b(?!['\u2019]))"
)

# Imperatives addressed to the agent are the user asking for work, not telling
# us a fact.  Kept to the verbs that open a request in practice; "remember",
# "prefer", "use" are deliberately absent because they open durable statements.
_LEADING_IMPERATIVE = re.compile(
    r"(?i)^\s*(?:list|show|display|print|dump|enumerate|count|give|tell|"
    r"fetch|search|find|check|run|rerun|re-?run|open|read|scan|inspect|"
    r"explain|summari[sz]e|recap|remind|retry|continue|proceed|"
    r"fix|debug|install|deploy|upgrade|restart)\b"
)

# Assistant status lines and canned failure text: re-derivable from the run that
# produced them, and actively harmful in recall ("your request was not
# processed" answers nothing).  Marker list is short on purpose — match only
# when the whole message is short enough to be mostly that sentence.
_FALLBACK_MARKERS = (
    "request was not processed",
    "was not processed",
    "please try again",
    "please resend",
    "something went wrong",
    "an error occurred",
    "operation interrupted",
    "operation was interrupted",
    "no results found",
    "could not complete",
    "couldn't complete",
    "cannot complete",
    "can't complete",
    "not able to complete",
    "i don't have access to",
    "i do not have access to",
    "not sure what you mean",
    "could you clarify",
    "could you confirm",
    "would you like me to",
    "let me know if you'd like",
    "let me know if you want",
    "as an ai",
    "as a language model",
)
MAX_FALLBACK_CHARS = 400

_STATUS_LINE_RE = re.compile(
    r"(?i)^\s*(?:re-?running|retrying|retry|attempting|trying again|"
    r"working on|processing|executing|loading|waiting|checking)\b"
)

# Gerund openers ("Running the migration…") carry facts as often as status
# ("Running the retention pass archives rather than deletes"), so they only
# count as narration alongside an explicit "now"/"this will take" marker.
_PROGRESS_OPENER_RE = re.compile(
    r"(?i)^\s*(?:running|starting|building|compiling|testing|deploying|"
    r"installing|downloading|uploading|fetching|generating)\b"
)
_NOW_MARKER_RE = re.compile(
    r"(?i)\b(?:now|moment|hold on|hang on|this will take|please wait|one sec|a second)\b"
)
MAX_PROGRESS_CHARS = 160


def is_request(text) -> bool:
    """True when `text` asks for something instead of stating something.

    Short by design: a long message is a document, and a trailing "?" inside
    one is not enough evidence that the writer was only asking.  Length also
    protects root-cause notes that happen to open with "Can" or cite a question
    ("Can't reproduce: the path resolves from node_modules").
    """
    if not isinstance(text, str) or not text:
        return False
    stripped = text.strip()
    if len(stripped) > MAX_REQUEST_CHARS:
        return False
    core = stripped.rstrip(" \t\n\"'`)\u201d]}")
    body = core.rstrip(".!")
    if core.endswith("?"):
        return len(core) <= MAX_SHORT_QUESTION_CHARS or bool(_LEADING_INTERROGATIVE.match(body))
    if _LEADING_IMPERATIVE.match(body):
        return True
    return len(body) <= MAX_UNMARKED_QUESTION_CHARS and bool(_LEADING_INTERROGATIVE.match(body))


def looks_like_fallback(text) -> bool:
    """True for assistant status narration and canned failure boilerplate."""
    if not isinstance(text, str) or not text:
        return False
    stripped = text.strip()
    lowered = stripped.lower()
    if len(stripped) <= MAX_FALLBACK_CHARS and any(m in lowered for m in _FALLBACK_MARKERS):
        return True
    if len(stripped) < MAX_REQUEST_CHARS and _STATUS_LINE_RE.match(stripped):
        return True
    return bool(
        len(stripped) <= MAX_PROGRESS_CHARS
        and _PROGRESS_OPENER_RE.match(stripped)
        and _NOW_MARKER_RE.search(stripped)
    )


def classify_candidate(text, role=None) -> str | None:
    """Classify candidate text into a memory type, or None when not worth storing.

    `role` is the author of the turn (`user`/`assistant`) when known.  It only
    sharpens two calls that content alone cannot make: an assistant turn must
    not be recorded as a *preference* (preferences are what the user wants), and
    assistant boilerplate is only boilerplate from the assistant.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if len(stripped) < 20:
        return None
    if len(stripped) > MAX_CANDIDATE_CHARS:
        return None
    if is_degenerate(stripped):
        return None
    if is_secret(stripped):
        return None
    if looks_injected(stripped):
        return None
    if is_request(stripped):
        return None
    if role == "assistant" and looks_like_fallback(stripped):
        return None
    if _is_transient(stripped):
        return None
    for memory_type, pattern in _CATEGORY_PATTERNS:
        if pattern.search(stripped):
            if memory_type == "preference" and role == "assistant":
                return "fact"
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
        memory_type = classify_candidate(text, role=role)
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
