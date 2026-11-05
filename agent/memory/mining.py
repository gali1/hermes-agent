"""Turn mining: secret rejection, transient filtering, and classification.

Ported from the OpenCode/MemPalace Hindsight primitives.  Stdlib only and
side-effect free: strips private blocks and injected context tags, rejects
secret-bearing and transient text, and classifies the remainder into a memory
type.
"""

from __future__ import annotations

import re
from collections import deque

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
# detection produce noise.  Raised from 3 KB after a shadow run over this store
# showed genuine findings being clipped as if they were dumps (a 4 KB PID/
# PostgreSQL root-cause analysis, a multi-KB library-convention note).  Prompt
# dumps are now caught by shape - task-brief and announcement openers - so
# length no longer has to carry that weight.
MAX_CANDIDATE_CHARS = 8000

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
# Reply openers: "Yes, finish …" / "Ok, run it" are the same request with a
# conversational prefix. Without this, a reply-shaped imperative is stored as a
# fact and user turns (importance 0.6) are the tier retention never retires.
_REPLY_FILLER_RE = re.compile(
    r"(?i)^\s*(?:yes|yeah|yep|no|nope|ok|okay|sure|alright|right|cool|perfect|"
    r"great|thanks|and|so|but|now|then|please|also|just)[,!.:\s]+\s*"
)

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
    r"(?i)^\s*(?:let\s+me|list|show|display|print|dump|enumerate|count|give|tell|"
    r"fetch|search|find|check|run|rerun|re-?run|open|read|scan|inspect|"
    r"explain|summari[sz]e|recap|remind|retry|continue|proceed|finish|"
    r"fix|debug|install|deploy|upgrade|restart)\b"
)

# "add / remove / delete / make / enable / change" are deliberately absent from
# the list above: as request verbs they also swallow standing directives that
# encode a durable decision ("Remove 50MB limit for telegram bot uploads/downloads",
# "Delete backup and regenerate the backup once changes have been made").
# They are caught at the other end instead — an action narrated while it happens
# opens with "Now", which is the shape the narration rule below handles.

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

# First-person work narration: "Now adding the algorithmic purge pass to the
# store:" is a move being narrated, not a finding, and it survived every other
# filter (it passed the 20-char floor and matched no category).
_MAX_NARRATION_CHARS = 200
_NARRATION_RE = re.compile(
    r"(?i)^\s*now\s+(?:adding|updating|writing|patching|editing|wiring|hooking|"
    r"extending|fixing|refactoring|rewriting|moving|tweaking|implementing|"
    r"checking|running|reading|searching"
    # "Now <imperative>" is the same move narrated in the present tense; the
    # request verbs that are too broad to ban outright (add/remove/delete) are
    # safe here because "Now" pins them to an action in progress.
    r"|add|remove|delete|rename|replace|insert|drop|update|edit|patch|wire|hook"
    r"|extend|fix|refactor|rewrite|tweak|implement|check|run|read|search|verify"
    r"|test|build|deploy|commit|push)\b"
)


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
    # A conversational prefix does not change what the sentence is doing.
    body = _REPLY_FILLER_RE.sub("", body) or body
    if _LEADING_IMPERATIVE.match(body):
        return True
    return len(body) <= MAX_UNMARKED_QUESTION_CHARS and bool(_LEADING_INTERROGATIVE.match(body))


# -- Narration lead-in (added) ------------------------------------------------
# The explicit "Now <verb>ing" form above misses the same move when a row does
# not open with "Now": "Adding the algorithmic purge pass to the store:" is
# narration too.  A gerund opener counts when the line is a lead-in - a
# colon-terminated header, or a short clause that asserts nothing.  Broad verbs
# (running/checking/reading) stay out of this path: they open findings as often
# as they open narration.
_MAX_NARRATION_LEADIN_CHARS = 80
_NARRATION_GERUND_RE = re.compile(
    r"(?i)^\s*(?:adding|updating|writing|patching|editing|wiring|hooking|extending|"
    r"refactoring|rewriting|tweaking|adjusting|implementing)\b"
)
_NARRATION_ASSERT_RE = re.compile(
    r"(?i)\b(?:means|requires|because|since|is|are|was|were|has|have|must|always|never)\b"
)


def looks_like_narration(text) -> bool:
    """True when `text` is the agent narrating its own next move."""
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > _MAX_NARRATION_CHARS:
        return False
    if _NARRATION_RE.match(stripped):
        return True
    if not _NARRATION_GERUND_RE.match(stripped):
        return False
    if stripped.endswith(":"):
        return True
    return (
        len(stripped) <= _MAX_NARRATION_LEADIN_CHARS
        and not _NARRATION_ASSERT_RE.search(stripped)
    )


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
    if looks_like_narration(stripped):
        return True
    return bool(
        len(stripped) <= MAX_PROGRESS_CHARS
        and _PROGRESS_OPENER_RE.match(stripped)
        and _NOW_MARKER_RE.search(stripped)
    )


# -- Task briefs and completion announcements --------------------------------
# Both are shaped like findings but are not: a brief is the prompt that started
# a run, an announcement is that run reporting it finished.  Session history
# holds both, so storing them adds recall noise that later reads as established
# fact ("Delivered. message id 5188" answers nothing a month later).
MAX_BRIEF_CHARS = 8000
_TASK_BRIEF_RE = re.compile(
    r"(?i)^\s*(?:-{2,}\s*)?(?:\*\*)?#*\s*(?:task|mission|objective|assignment|goal)\s*:"
)

# Announcements are completion *formulas*; the payload decides.  A formula with
# nothing after it, or followed by a report about the delivery itself ("message
# id 5188", "108 MB, 46.8s", "in the group", "nothing else pending"), is noise.
# A formula followed by a cause or a continued clause is a finding wearing a
# completion opener: "Fixed and verified. Root cause was NODE_OPTIONS…" and
# "Done — and the log you pasted exposed a second, live attack" both say
# something about the world.  Bare "verified" is deliberately NOT a completion
# formula: "Verified." introduces evidence ("Verified. Only `settings.json` was
# changed — 19 `favorite_models`…"), while "Fixed and verified." reports that
# work finished.  That distinction is lexical, not statistical — delivery
# reports are dense with ids, sizes and durations, so density cannot separate
# them from findings; both are specific.
MAX_ANNOUNCE_CHARS = 600
_ANNOUNCE_RE = re.compile(
    r"(?i)^\s*(?:\*\*)?\s*(?:delivered|fixed and verified|done|stopped|stopping|"
    r"complete|completed|finished|all set|all done|ready|shipped|reverted|rolled back)\b"
)

# Report about the delivery rather than about the world.
_ANNOUNCE_REPORT_RE = re.compile(
    r"(?i)\b(?:message[ _]?ids?|msg[ _]?ids?|chat[ _]?ids?|in the (?:channel|group|chat)|"
    r"both chats?|nothing else pending|no further action|as requested)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:MB|KB|GB|ms)\b|\b\d+(?:\.\d+)?\s*s\b"
)

# The clause that turns a completion formula into a finding.
_ANNOUNCE_PAYLOAD_RE = re.compile(
    r"(?i)\b(?:because|since|so that|root cause|due to|came from|turned out|"
    r"which (?:means|is why)|is held by|is (?:set|configured|owned) by)\b"
    r"|(?:[\u2014\u2013-]\s*and|,\s*and(?!\s*(?:nothing|no)\b)|,?\s*which\b|,\s*but\b)"
    r"|;\s*(?!nothing\b|no\b)"
    r"|\bport\s+\d{2,5}\b"
    r"|:"
)


def looks_like_brief(text) -> bool:
    """True when `text` is the brief that started a run, not knowledge."""
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > MAX_BRIEF_CHARS:
        return False
    return bool(_TASK_BRIEF_RE.match(stripped))


def looks_like_announcement(text) -> bool:
    """True when `text` is a completion formula with no finding behind it.

    The formula alone is not enough: "Verified. Only `settings.json` was
    changed — 19 `favorite_models`…" is evidence, so it is knowledge. So is
    "Fixed and verified. Root cause was NODE_OPTIONS…". Only a formula followed
    by a delivery report, or by nothing, is noise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > MAX_ANNOUNCE_CHARS:
        return False
    if not _ANNOUNCE_RE.match(stripped):
        return False
    if _ANNOUNCE_REPORT_RE.search(stripped):
        return True
    return not _ANNOUNCE_PAYLOAD_RE.search(stripped)


# -- Drop telemetry ----------------------------------------------------------
# Every rejection is recorded in a bounded in-memory ring - no I/O, no tokens.
# Silent filtering is how a good rule quietly starts eating real memories; this
# makes the drop set inspectable without changing any caller's signature.
_DROP_LOG: deque = deque(maxlen=200)


def _drop(reason: str, text: str):
    _DROP_LOG.append({"reason": reason, "chars": len(text), "preview": text[:120]})
    return None


def recent_drops(limit: int = 20) -> list[dict]:
    """Most recent rejections, newest last - for audit and shadow runs."""
    return list(_DROP_LOG)[-limit:]


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
        return _drop("too_short", stripped)
    if len(stripped) > MAX_CANDIDATE_CHARS:
        return _drop("too_long", stripped)
    if is_degenerate(stripped):
        return _drop("degenerate", stripped)
    if is_secret(stripped):
        return _drop("secret", stripped)
    if looks_injected(stripped):
        return _drop("injected", stripped)
    if looks_like_brief(stripped):
        return _drop("task_brief", stripped)
    if is_request(stripped):
        return _drop("request", stripped)
    if looks_like_announcement(stripped):
        return _drop("announcement", stripped)
    if role == "assistant" and looks_like_fallback(stripped):
        return _drop("fallback", stripped)
    if _is_transient(stripped):
        return _drop("transient", stripped)
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
