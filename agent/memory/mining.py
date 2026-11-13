"""Turn mining: secret rejection, transient filtering, and classification.

Ported from the OpenCode/MemPalace Hindsight primitives.  Stdlib only and
side-effect free: strips private blocks and injected context tags, rejects
secret-bearing and transient text, and classifies the remainder into a memory
type.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter, deque

from agent.memory.dedup import is_degenerate, normalize_content

logger = logging.getLogger(__name__)

# Bumped whenever a filter's decision on existing text can change.  Stored on
# every captured row so audits can re-check exactly the rows an older version let
# through, instead of re-reading the whole store.
CAPTURE_FILTER_VERSION = 2

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

# -- First-person narration --------------------------------------------------
# "Now adding the algorithmic purge pass to the store:" is a move being
# narrated, not a finding, and it survived every other filter (it passed the
# 20-char floor and matched no category).  Kept separate from
# _LEADING_IMPERATIVE on purpose: it is not the user asking for work, and a
# distinct drop reason makes the telemetry readable.
# Length-capped for the same reason as announcements - a long passage that
# happens to open this way usually carries real content.
MAX_NARRATION_CHARS = 200
_MAX_NARRATION_LEADIN_CHARS = 80
# "let me" and "now <verb>ing" are unambiguous: the words pin the sentence to an
# action in progress.  Kept alongside the explicit "Now <verb>" form below so a
# narrated move is caught whichever tense it is written in.
_NARRATION_EXPLICIT_RE = re.compile(r"(?i)^\s*(?:now\s+)?let me\b|^\s*now\s+[a-z]+ing\b")
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

# A gerund opener is only narration when the line is a lead-in - either a
# colon-terminated header ("Adding the purge pass to the store:") or a short
# clause that asserts nothing ("Updating the config to point at the new
# endpoint.").  Broad verbs like running/checking/reading open findings as often
# as narration ("Running the retention pass archives rather than deletes"), so
# they only count with an explicit "now".
_NARRATION_GERUND_RE = re.compile(
    r"(?i)^\s*(?:adding|updating|writing|patching|editing|wiring|hooking|extending|"
    r"refactoring|rewriting|tweaking|adjusting|implementing)\b"
)
_NARRATION_ASSERT_RE = re.compile(
    r"(?i)\b(?:means|requires|because|since|is|are|was|were|has|have|must|always|never)\b"
)


# -- Durable-statement guard and conversational lead-ins (v2) -----------------
# A standing instruction can open exactly like a one-off request ("Run tests
# with -x, always"; "Can you always answer in English?").  The words that make
# an instruction durable are few and explicit, so their presence vetoes every
# request/probe drop below.  Safety drops (secret, injected) are not vetoed.
_DURABLE_RE = re.compile(
    r"(?i)\b(?:always|never|from now on|going forward|henceforth|in (?:the )?future|"
    r"by default|whenever|every time|each time|any ?time|remember (?:that|to)|"
    r"prefer\w*|don'?t ever|do not ever|must(?:n'?t| not)?|unless|"
    r"standing (?:rule|instruction|order))\b"
)

# Chat lead-ins carry no meaning but hide the real opener from the anchored
# patterns: "just copy the files\u2026", "ok now restart it", "please list \u2026".
_LEADING_FILLER_RE = re.compile(
    r"(?i)^\s*(?:(?:ok(?:ay)?|so|now|then|just|please|pls|plz|hey|alright|and|also|"
    r"well|right|yes|yeah|yep|great|cool|thanks|thx|hmm|um|oh)\b[,.!:;\-\u2014]*\s+)+"
)
# "Can you \u2026" without a question mark is still an ask; the remainder is then
# judged like an imperative so the durable guard applies to it.
_POLITE_ASK_RE = re.compile(
    r"(?i)^\s*(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+|kindly\s+|just\s+)?"
)

# Imperatives that are ambiguous on their own ("remove the schedule fleet
# maintenance" is a standing directive) and only count as a one-off request
# when the sentence also leans on the conversation around it (deixis such as
# "mentioned earlier", "that", "again") or names a specific agent tool.  This is
# what lets "just copy the files mentioned earlier into the location only" go
# without the over-reach the skill recorded for a bare verb list.
_CONTEXTUAL_IMPERATIVE_RE = re.compile(
    r"(?i)^(?:copy|move|rename|delete|remove|add|make|create|send|update|change|set|"
    r"use|try|do|redo|undo|apply|switch|put|write|edit|compare|verify|test|"
    r"analy[sz]e|review|look|go|stop|kill|start|enable|disable|clean|prune|save|"
    r"load|merge|revert|push|pull|commit|build|convert|download|upload|sync|mount|"
    r"unmount|reboot|turn|keep|drop|replace|repeat|resume|cancel|post|reply|answer)\b"
)
_DEICTIC_RE = re.compile(
    r"(?i)\b(?:earlier|above|below|previous(?:ly)?|mentioned|aforementioned|those|these|"
    r"them|again|same ones?|the rest|as discussed|you (?:just|said)|i (?:just )?(?:sent|pasted|gave))\b"
)
# Agent tool names: the request is about one turn's work, never a durable fact.
_TOOL_REF_RE = re.compile(
    r"\b(?:web_search|web_extract|execute_code|terminal|read_file|write_file|patch|"
    r"search_files|session_search|skill_view|skill_manage|skills_list|delegate_task|"
    r"send_message|todo|clarify|vision_analyze|image_generate|browser_[a-z_]+|"
    r"mcp_[a-z0-9_]+|[a-z]+_(?:search|extract|tool))\b"
)


def _request_body(text: str) -> tuple[str, bool]:
    """Strip lead-ins; report whether a polite "can you" ask was removed."""
    body = _LEADING_FILLER_RE.sub("", text, count=1)
    polite = _POLITE_ASK_RE.match(body)
    if polite:
        body = body[polite.end():]
    return body.strip(), bool(polite)


def is_request(text) -> bool:
    """True when `text` asks for something instead of stating something.

    Short by design: a long message is a document, and a trailing "?" inside
    one is not enough evidence that the writer was only asking.  Length also
    protects root-cause notes that happen to open with "Can" or cite a question
    ("Can't reproduce: the path resolves from node_modules").

    v2: lead-ins ("just", "ok now", "please") no longer hide the opener; a
    polite "can you \u2026" is judged as an imperative; ambiguous verbs count only
    with conversational deixis or a tool name; and any durable marker
    ("always", "from now on", "unless") vetoes the imperative paths.
    """
    if not isinstance(text, str) or not text:
        return False
    stripped = text.strip()
    if len(stripped) > MAX_REQUEST_CHARS:
        return False
    core = stripped.rstrip(" \t\n\"'`)\u201d]}")
    body, polite = _request_body(core.rstrip(".!"))
    durable = bool(_DURABLE_RE.search(stripped))
    if polite:
        if durable:
            return False
        return True
    if core.endswith("?"):
        return len(core) <= MAX_SHORT_QUESTION_CHARS or bool(_LEADING_INTERROGATIVE.match(body))
    # A conversational prefix does not change what the sentence is doing
    # ("Yes finish and make it automated…" is still an ask).
    body = _REPLY_FILLER_RE.sub("", body) or body
    # A durable marker vetoes every imperative path, whichever prefix hid it.
    if durable:
        return False
    if _LEADING_IMPERATIVE.match(body):
        return True
    if _CONTEXTUAL_IMPERATIVE_RE.match(body) and (
        _DEICTIC_RE.search(body) or _TOOL_REF_RE.search(body)
    ):
        return True
    return len(body) <= MAX_UNMARKED_QUESTION_CHARS and bool(_LEADING_INTERROGATIVE.match(body))


def looks_like_narration(text) -> bool:
    """True when `text` is the agent narrating its own next move."""
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > MAX_NARRATION_CHARS:
        return False
    if _NARRATION_EXPLICIT_RE.match(stripped) or _NARRATION_RE.match(stripped):
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


# -- Protocol probes (v2) ----------------------------------------------------
# Connectivity and harness tests: "Reply with exactly: MEMOK", "Reply with the
# single word: ok", relay handshakes between Hermes hosts.  They describe a
# message, not the world, and they are shaped like imperatives, so on recall they
# read as standing instructions.  The tail anchor (a single token, optionally
# quoted, then end of text) is what separates a probe from a durable formatting
# rule such as "Reply with exactly one sentence on Telegram".
MAX_PROBE_CHARS = 160
MAX_HANDSHAKE_CHARS = 400
MAX_SENTINEL_CHARS = 60
_PROBE_TOKEN = r"[:\-—]?\s*['\"“‘`]?[\w.\-!]{1,40}['\"”’`]?[.!]?"
_PROBE_OPENER_RE = re.compile(
    r"(?i)^(?:reply|respond|answer|say|output|print|return|echo|type|write|send)\b"
    r"[^\n]{0,40}?\b(?:exactly|only|just|verbatim|literally|nothing but|"
    r"the (?:single |one )?word|one word)\s*" + _PROBE_TOKEN + r"\s*$"
)
# A handshake can sit at the end of a longer relay message ("Hank here, from the
# other Hermes box … reply with exactly 'relay-7f3a'"), so it is matched anywhere
# but needs a quoted token, which a prose rule never has.
_HANDSHAKE_RE = re.compile(
    r"(?i)\b(?:reply|respond|answer|ack(?:nowledge)?)\s+(?:back\s+)?(?:with|using)\s+"
    r"(?:exactly|only|just|the (?:single |one )?word|the (?:token|code|string))\s*[:\-]?\s*"
    r"['\"“‘`][^'\"”’`\s]{1,40}['\"”’`]"
)
# Bare sentinels the far side echoes back: "PONG", "DONEALL FINALOK".
_SENTINEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_\-]+(?:[ \t]+[A-Z0-9][A-Z0-9_\-]+){0,3}[.!]?$")
_PING_RE = re.compile(
    r"(?i)^(?:ping|pong|test(?:ing)?(?: message)?|are you (?:there|alive|up|awake)|"
    r"health ?check|connectivity (?:test|check)|can you hear me)\b[^\n]{0,40}$"
)


def looks_like_probe(text) -> bool:
    """True for harness/connectivity probes and the sentinels that answer them."""
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) <= MAX_SENTINEL_CHARS and (
        _SENTINEL_RE.match(stripped) or _PING_RE.match(stripped)
    ):
        return True
    if len(stripped) <= MAX_HANDSHAKE_CHARS and _HANDSHAKE_RE.search(stripped):
        return True
    if len(stripped) > MAX_PROBE_CHARS or _DURABLE_RE.search(stripped):
        return False
    body, _polite = _request_body(stripped)
    return bool(_PROBE_OPENER_RE.match(body))


# -- Tool output renderings (v2) ---------------------------------------------
# "Last 10 sessions (173 total in state.db)…", "84 skills across 15 categories:
# …", "Oldest entry in the memory store (…):" and pasted tables are a rendering
# of tool output.  They are regenerable and stale the moment they are stored.
# Detection uses the *raw* text with its line structure: the old pipeline
# collapsed whitespace before classifying, which made tables invisible.
_DUMP_HEADER_RE = re.compile(
    r"(?i)^\s*(?:#+\s*|\*\*|>\s*)?(?:"
    r"(?:last|first|top|recent|latest|oldest|newest|next)\s+\d[\d,]*\s+[a-z]"
    r"|\d[\d,]*\s+(?:[a-z][\w\-]*\s+){0,3}(?:across|found|total|matching|returned|"
    r"listed|in total|available)\b"
    r"|(?:oldest|newest|latest|most recent)\s+(?:entry|entries|row|rows|record|records|"
    r"memor(?:y|ies)|sessions?|messages?|items?|results?|jobs?|files?)\b"
    r"|here(?:'s| is| are)\s+(?:the\s+)?(?:full\s+|complete\s+|current\s+)?"
    r"(?:list|output|results?|table|dump|inventory|listing|breakdown)\b"
    r"|(?:results?|output|listing|inventory|stdout|stderr)\s*:"
    r")"
)
# A header alone is not enough ("3 nodes in the cluster run Debian" is a fact):
# it must introduce something - a colon or parenthesis on the header line, or a
# structured body underneath it.
_DUMP_HEADER_INTRO_RE = re.compile(r"^[^\n]{0,160}?[:(]")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_STRUCT_LINE_RE = re.compile(
    r"^\s*(?:[-*•▪◦+]\s|\d{1,3}[.)]\s|\|"
    r"|[\w .\-/()]{1,40}:\s+\S"
    r"|[0-9a-f]{8,}\b"
    r"|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})"
)
_BOX_DRAWING_RE = re.compile(r"[─-╿]{3,}")
_PROSE_LINE_RE = re.compile(r"(?:\w+\W+){7,}\w+[.!?]['\")\]]?\s*$")
MIN_DUMP_LINES = 6
DUMP_STRUCTURED_SHARE = 0.7
DUMP_PROSE_SHARE = 0.3
# A list flattened onto one line: "claude-code — Delegate … codex — …".
MIN_COLLAPSED_SEPARATORS = 8


def looks_like_tool_dump(text, raw=None) -> bool:
    """True when `text` renders tool output (table, listing, inventory)."""
    if not isinstance(text, str) or not text.strip():
        return False
    source = raw if isinstance(raw, str) and raw.strip() else text
    lines = [line for line in source.splitlines() if line.strip()]
    head = lines[0] if lines else ""
    if any(_BOX_DRAWING_RE.search(line) for line in lines[:50]):
        return True
    structured = sum(1 for line in lines if _STRUCT_LINE_RE.match(line))
    if _DUMP_HEADER_RE.match(head) and (
        _DUMP_HEADER_INTRO_RE.match(head) or structured >= 3
    ):
        return True
    if sum(1 for line in lines if _TABLE_ROW_RE.match(line)) >= 3:
        return True
    if len(lines) >= MIN_DUMP_LINES:
        prose = sum(1 for line in lines if _PROSE_LINE_RE.search(line))
        if (structured / len(lines) >= DUMP_STRUCTURED_SHARE
                and prose / len(lines) < DUMP_PROSE_SHARE):
            return True
    flat = text.strip()
    return flat.count(" — ") + flat.count(" -- ") >= MIN_COLLAPSED_SEPARATORS


# -- Tool echo (v2) ----------------------------------------------------------
# The assistant often restates what a tool just returned ("Oldest entry in the
# memory store: …").  That restatement is as regenerable as the tool output, so
# it is matched against the turn's own tool results by word shingles.
_ECHO_SHINGLE = 6
MIN_ECHO_SHINGLES = 4
ECHO_SHARE = 0.6
MAX_TOOL_TEXT_CHARS = 400_000
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def text_shingles(text, size: int = _ECHO_SHINGLE) -> frozenset:
    """Hashed word n-grams of `text` (lower-cased); empty for short text."""
    if not isinstance(text, str) or not text:
        return frozenset()
    words = _WORD_RE.findall(text.lower())
    if len(words) < size:
        return frozenset()
    return frozenset(hash(tuple(words[i:i + size])) for i in range(len(words) - size + 1))


def echo_share(text, tool_shingles) -> float:
    """Share of `text`'s shingles that also occur in the tool output."""
    if not tool_shingles:
        return 0.0
    own = text_shingles(text)
    if len(own) < MIN_ECHO_SHINGLES:
        return 0.0
    return len(own & tool_shingles) / len(own)


# -- Operator patterns (v2) --------------------------------------------------
# config.yaml ``memory.enhanced.capture.drop_patterns`` / ``keep_patterns``: the
# operator extends the gate without a code change.  ``keep`` wins over every
# heuristic drop but never over a safety drop (secret, injected, size bounds).
_PATTERN_LOCK = threading.Lock()
_CUSTOM_DROP: tuple = ()
_CUSTOM_KEEP: tuple = ()


def configure_capture(drop_patterns=None, keep_patterns=None) -> list[str]:
    """Install operator regexes; returns one error string per invalid pattern.

    Invalid patterns are skipped (logged), never raised: a typo in config.yaml
    must not take the memory layer down.
    """
    global _CUSTOM_DROP, _CUSTOM_KEEP
    errors: list[str] = []

    def _compile(patterns) -> tuple:
        compiled = []
        for pattern in patterns or ():
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                errors.append(f"{pattern!r}: {exc}")
                logger.warning("memory capture pattern ignored (%s): %r", exc, pattern)
        return tuple(compiled)

    drop, keep = _compile(drop_patterns), _compile(keep_patterns)
    with _PATTERN_LOCK:
        _CUSTOM_DROP, _CUSTOM_KEEP = drop, keep
    return errors


# -- Dispositions ------------------------------------------------------------
# "drop" discards; "quarantine" stores the row with status='quarantined', which
# recall never reads but audit/restore can.  Quarantine is for the classes where
# a false positive would cost a real finding (big blobs, tool-shaped text), so a
# miss is recoverable instead of silent.
QUARANTINE_REASONS = frozenset({"too_long", "tool_dump", "tool_echo"})
# Reasons precise enough to act on without a human: used by the scheduled audit
# and by the recall guard that keeps legacy junk out of the prompt.
HIGH_PRECISION_REASONS = frozenset({"secret", "injected", "probe", "tool_dump", "task_brief"})


# -- Drop telemetry ----------------------------------------------------------
# Every rejection is recorded in a bounded in-memory ring - no I/O, no tokens.
# Silent filtering is how a good rule quietly starts eating real memories; this
# makes the drop set inspectable without changing any caller's signature.
_DROP_LOG: deque = deque(maxlen=200)
_DROP_COUNTS: Counter = Counter()
_TELEMETRY_LOCK = threading.Lock()


def _drop(reason: str, text: str):
    with _TELEMETRY_LOCK:
        _DROP_LOG.append({"reason": reason, "chars": len(text), "preview": text[:120]})
        _DROP_COUNTS[reason] += 1
    return None


def recent_drops(limit: int = 20) -> list[dict]:
    """Most recent rejections, newest last - for audit and shadow runs."""
    with _TELEMETRY_LOCK:
        return list(_DROP_LOG)[-limit:]


def drop_stats() -> dict:
    """Rejections per reason since process start (drops and quarantines)."""
    with _TELEMETRY_LOCK:
        return dict(_DROP_COUNTS)


def _verdict(verdict: str, reason: str, memory_type=None) -> dict:
    return {"verdict": verdict, "reason": reason, "memory_type": memory_type,
            "filter_version": CAPTURE_FILTER_VERSION}


def assess_candidate(text, role=None, *, raw=None, tool_shingles=None,
                     record: bool = True) -> dict:
    """Full capture decision: ``{"verdict", "reason", "memory_type", "filter_version"}``.

    ``verdict`` is ``keep`` (store active), ``quarantine`` (store, never recall)
    or ``drop``.  ``raw`` is the text before whitespace collapsing, so table and
    list shape stay visible; ``tool_shingles`` (from :func:`text_shingles`) is the
    turn's tool output, used to catch the assistant restating it.  ``record``
    false skips telemetry (audits, recall guard).  Pure regex, no I/O, no model.
    """
    def _reject(reason: str) -> dict:
        verdict = "quarantine" if reason in QUARANTINE_REASONS else "drop"
        if record:
            _drop(reason, stripped)
        return _verdict(verdict, reason)

    if not isinstance(text, str):
        return _verdict("drop", "not_text")
    stripped = text.strip()
    if len(stripped) < 20:
        return _reject("too_short")
    if len(stripped) > MAX_CANDIDATE_CHARS:
        return _reject("too_long")
    if is_degenerate(stripped):
        return _reject("degenerate")
    if is_secret(stripped):
        return _reject("secret")
    if looks_injected(stripped):
        return _reject("injected")

    with _PATTERN_LOCK:
        custom_drop, custom_keep = _CUSTOM_DROP, _CUSTOM_KEEP
    kept_by_operator = any(pattern.search(stripped) for pattern in custom_keep)
    if not kept_by_operator:
        if any(pattern.search(stripped) for pattern in custom_drop):
            return _reject("operator_pattern")
        if looks_like_probe(stripped):
            return _reject("probe")
        if looks_like_brief(stripped):
            return _reject("task_brief")
        if looks_like_tool_dump(stripped, raw):
            return _reject("tool_dump")
        if (role == "assistant" and tool_shingles
                and echo_share(stripped, tool_shingles) >= ECHO_SHARE):
            return _reject("tool_echo")
        # Narration is checked before the request rule: "Let me …" is both an
        # imperative and a narrated move, and the distinct drop reason is what
        # makes the telemetry readable.
        if looks_like_narration(stripped):
            return _reject("narration")
        if is_request(stripped):
            return _reject("request")
        if looks_like_announcement(stripped):
            return _reject("announcement")
        if role == "assistant" and looks_like_fallback(stripped):
            return _reject("fallback")
        if _is_transient(stripped):
            return _reject("transient")
    for memory_type, pattern in _CATEGORY_PATTERNS:
        if pattern.search(stripped):
            if memory_type == "preference" and role == "assistant":
                return _verdict("keep", "kept", "fact")
            return _verdict("keep", "kept", memory_type)
    return _verdict("keep", "kept", "fact")


def classify_candidate(text, role=None) -> str | None:
    """Classify candidate text into a memory type, or None when not worth storing.

    `role` is the author of the turn (`user`/`assistant`) when known.  It only
    sharpens two calls that content alone cannot make: an assistant turn must
    not be recorded as a *preference* (preferences are what the user wants), and
    assistant boilerplate is only boilerplate from the assistant.

    Compatibility wrapper over :func:`assess_candidate`: quarantine counts as
    "not worth storing" here, because callers of this function store what it
    returns as active memory.
    """
    verdict = assess_candidate(text, role=role)
    return verdict["memory_type"] if verdict["verdict"] == "keep" else None


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


def _tool_shingles(messages, tool_texts) -> frozenset:
    """Shingles of the tool output to compare assistant text against.

    Explicit ``tool_texts`` (the current turn's results, supplied by the backend)
    win; otherwise the ``role: tool`` messages inside ``messages`` are used, which
    covers session-end and pre-compression mining over whole transcripts.  Capped
    at ``MAX_TOOL_TEXT_CHARS`` of the most recent output.
    """
    if tool_texts is None:
        tool_texts = [
            _content_text(m.get("content")) for m in messages or ()
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
    budget, parts = MAX_TOOL_TEXT_CHARS, []
    for chunk in reversed([t for t in tool_texts or () if isinstance(t, str) and t]):
        if budget <= 0:
            break
        parts.append(chunk[-budget:])
        budget -= len(parts[-1])
    return text_shingles("\n".join(reversed(parts))) if parts else frozenset()


def mine_turns(messages, *, min_length=20, limit=20, tool_texts=None,
               include_quarantined=False) -> list[dict]:
    """Mine OpenAI-style message dicts into deduplicated memory candidates.

    Each candidate is ``{"content", "memory_type", "role"}``.
    With ``include_quarantined`` the result also carries rows the gate wants kept
    out of recall, marked ``"quarantine": <reason>``; ``limit`` counts only kept
    rows, so quarantine can never crowd out real knowledge.
    """
    if not messages or limit <= 0:
        return []

    shingles = _tool_shingles(messages, tool_texts)
    mined = []
    kept = 0
    seen = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        raw = _content_text(message.get("content"))
        if not raw:
            continue
        raw = strip_private(raw)
        raw = _CONTEXT_TAGS_RE.sub(" ", raw)
        text = re.sub(r"\s+", " ", raw).strip()
        if not text or len(text) < min_length:
            continue
        if is_secret(text):
            continue
        normalized = normalize_content(text)
        if normalized in seen:
            continue
        verdict = assess_candidate(text, role=role, raw=raw, tool_shingles=shingles)
        if verdict["verdict"] == "drop":
            continue
        if verdict["verdict"] == "quarantine":
            if include_quarantined:
                seen.add(normalized)
                mined.append({"content": text, "memory_type": "fact", "role": role,
                              "quarantine": verdict["reason"]})
            continue
        seen.add(normalized)
        mined.append({"content": text, "memory_type": verdict["memory_type"], "role": role})
        kept += 1
        if kept >= limit:
            break
    return mined
