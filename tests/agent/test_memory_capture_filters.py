"""Capture filters for turn mining: requests, status lines, injected prompts.

Two sources feed these cases.  The request/fallback/injected/secret classes come
from the dev-box implementation of ``agent/memory/mining.py``; the brief,
announcement and drop-telemetry cases are local additions made after a shadow
run over this host's live store showed the 3 KB cap clipping genuine findings
while a "**Delivered.** …" status line survived as a fact.

Ground truth for the junk cases is real rows that reached the live store via
auto-capture (an interrogative, a meta-command, and a canned failure line);
ground truth for the signal cases is real root-cause notes that must survive.

Every filter here is pure regex - no model call, no tokens.
"""

import agent.memory.mining as mining
from agent.memory.mining import (
    classify_candidate,
    is_request,
    is_secret,
    looks_injected,
    looks_like_announcement,
    looks_like_brief,
    looks_like_fallback,
    mine_turns,
    recent_drops,
)


JUNK_USER = [
    "All enhanced memory features working?",
    "list all skills available",
    "List all of the skills you have available right now",
    "Is the gateway running on port 8080?",
    "did the test suite pass after the patch?",
    "Show me the contents of the memory database file",
    "Can you restart the gateway for me please?",
    "Hello! How can I help you today?",
    "Let me check the current store size before changing anything.",
    "Yes finish and make it automated using the algorithmic approach.",
    "Let me verify the final state of the settings after all changes.",
    "Now let me check it compiles, then build and deploy.",
]

JUNK_ASSISTANT = [
    "Your request was not processed. Send it again if you still want me to carry it out.",
    "Operation interrupted: the process was stopped before it finished.",
    "Please try again later — something went wrong on the provider side.",
    "Re-running the build with the patched mining module loaded.",
    "Would you like me to prune those three memories now?",
    "Could you confirm which provider you want to use before I continue?",
    "I don't have access to that directory from this session.",
    "Running the migration now, this will take a moment.",
    # Work narration that survived every earlier filter (passed the 20-char
    # floor, matched no category) and reached the live store as a fact.
    "Now adding the algorithmic purge pass to the store:",
]

SIGNAL_USER = [
    "I prefer using uv for Python environments and never pip install globally",
    "We decided to use SQLite for the local memory engine store",
    "The gateway runs on port 8080 behind the nginx proxy in production",
    # Standing directives: "add/remove/delete" are no longer request verbs, so a
    # policy change recorded as an instruction survives.
    "Remove 50MB limit for telegram bot uploads/downloads",
    "Delete backup and regenerate the backup once changes have been made",
    # Bug report, not a question: opens with "When" but is a claim about state.
    "When the user select vtcode_wsl and moves to the model list it only shows deepseek-v4-flash-free add the model",
]

SIGNAL_ASSISTANT = [
    "Can't reproduce the crash: the path resolves from node_modules, not the stale bashrc.",
    "The esbuild failure came from NODE_OPTIONS exported twice by an old bashrc line.",
    "Root cause: the ACP client sends its own cwd, which overrides the --cwd flag.",
    "Fixed by normalizing the Windows path to the home directory before the session opens.",
    # "Now" + a plain clause is a claim, not narration.
    "Now the store archives rather than deletes, so retention stays reversible.",
    # A reply opener does not turn a fact into a request.
    "Yes, the gateway is behind nginx and owns the websocket transport.",
]


def test_request_detection_on_real_junk():
    for text in JUNK_USER:
        assert is_request(text), text


def test_request_detection_keeps_claims():
    for text in SIGNAL_USER + SIGNAL_ASSISTANT:
        assert not is_request(text), text


def test_fallback_detection_on_real_junk():
    for text in JUNK_ASSISTANT:
        assert looks_like_fallback(text), text


def test_fallback_detection_keeps_root_cause_notes():
    for text in SIGNAL_ASSISTANT + SIGNAL_USER:
        assert not looks_like_fallback(text), text


def test_classify_rejects_requests_and_boilerplate():
    for text in JUNK_USER:
        assert classify_candidate(text, role="user") is None, text
    for text in JUNK_ASSISTANT:
        assert classify_candidate(text, role="assistant") is None, text


def test_classify_keeps_signal():
    assert classify_candidate(SIGNAL_USER[0], role="user") == "preference"
    assert classify_candidate(SIGNAL_USER[1], role="user") == "decision"
    assert classify_candidate(SIGNAL_USER[2], role="user") == "fact"
    for text in SIGNAL_ASSISTANT:
        assert classify_candidate(text, role="assistant") is not None, text


def test_assistant_turn_is_never_a_preference():
    # Preferences describe the user; an assistant turn mentioning "prefer" is
    # narration and was previously stored as a standing preference.
    text = "The store prefers the RRF fusion order because it keeps recall stable"
    assert classify_candidate(text, role="assistant") == "fact"
    assert classify_candidate(text, role="user") == "preference"


def test_role_none_still_classifies():
    assert classify_candidate("I prefer dark mode in every editor I use") == "preference"
    assert classify_candidate("All enhanced memory features working?") is None


def test_long_text_with_question_mark_survives():
    text = (
        "Investigated the retention pass and confirmed it archives rather than deletes, "
        "writes a cold export first, and only then flips the row status — should the "
        "export raise, does it fail closed?"
    )
    assert not is_request(text)
    assert classify_candidate(text, role="assistant") is not None


def test_mine_turns_drops_requests_and_boilerplate():
    messages = [
        {"role": "user", "content": "All enhanced memory features working?"},
        {"role": "user", "content": "list all skills available"},
        {"role": "assistant", "content": "Your request was not processed. Send it again if you still want me to carry it out."},
        {"role": "assistant", "content": "Would you like me to prune those three memories now?"},
        {"role": "user", "content": "I prefer using uv for Python environments and never pip install globally"},
        {"role": "assistant", "content": "Root cause: the ACP client sends its own cwd, which overrides the CLI argument at session start."},
    ]
    mined = mine_turns(messages)
    contents = [m["content"] for m in mined]
    assert contents == [
        "I prefer using uv for Python environments and never pip install globally",
        "Root cause: the ACP client sends its own cwd, which overrides the CLI argument at session start.",
    ]
    assert {m["memory_type"] for m in mined} == {"preference", "fact"}


DROPPED_ANNOUNCEMENTS = [
    # Formula followed by a report about the delivery: ids, sizes, durations,
    # destination, or "nothing pending". Density of detail is what a delivery
    # report is made of, so detail cannot be the discriminator.
    "Delivered. message id 5188",
    "Delivered. The digest went out to both chats, message id 5188, 108 MB, 46.8s.",
    "Delivered. Updated `~/.hermes/config.yaml` and restarted; message id 5188.",
    "Fixed and verified. The two messages are in the group now.",
    "Done. The report is in the channel.",
    "All done — shipped.",
    "Completed the run; nothing else pending.",
    "Done.",
]

KEPT_ANNOUNCEMENTS = [
    # "Verified." opens onto evidence, so it is not a completion formula at all.
    "Verified. Only `settings.json` was changed — 19 `favorite_models` for the opencode provider.",
    "Verified — `settings.json` has 5 `available_models`, 19 `favorite_models`, and clean bare-model names",
    "Verified. The ACP session crash came from a Windows cwd sent by Zed overriding --cwd.",
    # Formula plus a payload: a cause, a continued clause, a named state.
    "Fixed and verified. Root cause was NODE_OPTIONS exported twice by an old bashrc line.",
    "Done in a way that matters: `retire_stale` now writes a cold export before archiving.",
    "Shipped the retention fix; rows archive with a cold_export path in metadata.",
    "Done — and the log you pasted exposed a second, live attack.",
    "Fixed and verified the deploy, port 8080 answers again",
]


def test_announcement_is_a_formula_without_payload():
    for text in DROPPED_ANNOUNCEMENTS:
        assert looks_like_announcement(text), text
        assert classify_candidate(text, role="assistant") is None, text
    for text in KEPT_ANNOUNCEMENTS:
        assert not looks_like_announcement(text), text
        assert classify_candidate(text, role="assistant") is not None, text


def test_injected_wrappers_are_rejected():
    # Gateway-added wrappers reached the live store as memories; they are
    # instruction/injected context, never knowledge about the user's world.
    assert looks_injected(
        "[System note: The previous turn was interrupted by a gateway shutdown; the gateway is now back online.]"
    )
    assert looks_injected("Recalled memory (background evidence; validate against current state): - [fact] x")
    assert classify_candidate(
        "[System note: The previous turn was interrupted by a gateway shutdown; the gateway is now back online.]",
        role="user",
    ) is None


# -- requests: asks for something vs states something ------------------------

def test_short_question_is_request():
    assert is_request("Is plex running without issue?")
    assert is_request("list all sessions from today")
    assert is_request("Hank how is Plex running?")
    assert is_request("What do you mean \"a couple hours...\"?")


def test_imperative_to_agent_is_request():
    assert is_request("install zip via cli and use computer if prompted")
    assert is_request("restart the gateway")
    assert is_request("show me the disk usage")


def test_apostrophe_claim_is_not_request():
    # "Can't reproduce…" leads with an interrogative but asserts a finding.
    assert not is_request("Can't reproduce: the path resolves from node_modules")


def test_long_paragraph_with_question_is_not_request():
    body = (
        "The retention pass archives rather than deletes, which is why the vault "
        "kept growing for six weeks before anyone noticed. Does that match what "
        "you saw in the logs? " + "Additional context. " * 20
    )
    assert not is_request(body)


def test_durable_imperatives_are_kept():
    # "remove" is deliberately absent from the imperative list: it opens a
    # standing instruction, not a lookup.
    assert classify_candidate("remove the schedule fleet maintenance.") is not None


# -- assistant status / fallback boilerplate --------------------------------

def test_fallback_boilerplate_detected():
    assert looks_like_fallback("Your request was not processed. Please try again.")
    assert looks_like_fallback("Operation interrupted before completion.")
    assert looks_like_fallback("Something went wrong")


def test_status_and_progress_lines_detected():
    assert looks_like_fallback("Re-running the build now")
    assert looks_like_fallback("Loading the model, this will take a moment")
    assert looks_like_fallback("Retrying")


def test_progress_opener_without_now_marker_is_kept():
    # "Running the retention pass archives rather than deletes" is knowledge.
    assert not looks_like_fallback(
        "Running the retention pass archives rather than deletes, which is why growth looked linear"
    )


# -- injected prompt text ---------------------------------------------------

def test_injected_markers_detected():
    assert looks_injected("[IMPORTANT: Background process proc_abc completed normally")
    assert looks_injected("You are running as a scheduled cron job. DELIVERY: your response")
    assert looks_injected("Recalled memory (background evidence; validate against current state)")
    assert looks_injected("leftover <memory-context> fence")


def test_ordinary_prose_not_injected():
    assert not looks_injected("The gateway owns the websocket transport on port 8642")


# -- task briefs and completion announcements (local additions) -------------

def test_task_brief_detected():
    assert looks_like_brief("--- # Task: Safely Audit, Restructure, Verify and Report")
    assert looks_like_brief("# Mission: Diagnose and Fix Excessive PostgreSQL CPU Usage")
    assert looks_like_brief("**Objective:** reclaim the vault")


def test_announcement_detected_when_short():
    assert looks_like_announcement("**Delivered.** The full backup is in this chat")
    assert looks_like_announcement("**Fixed and verified.** Two messages sent")
    assert looks_like_announcement("Stopped. Background upload killed.")


def test_announcement_allows_long_finding():
    # Opens like a status line but carries a real root cause: length decides.
    text = (
        "Done - and the log you pasted was current, not stale. It exposed a "
        "second live attack: the miner binary was re-dropped after the first "
        "cleanup, and the watchdog missed it because the path changed. " * 6
    )
    assert not looks_like_announcement(text)


# -- size cap ---------------------------------------------------------------

def test_findings_under_new_cap_are_kept():
    # A 4 KB root-cause write-up must survive: the old 3 KB cap ate these.
    text = (
        "Signal resolved. PID 33236 was never PostgreSQL. Root cause confirmed: "
        "the process was a repackaged miner listening on 5432. " + "Detail. " * 700
    )
    assert len(text) > 3000 and len(text) < 8000
    assert classify_candidate(text) is not None


def test_oversized_dump_is_dropped():
    text = "System inventory follows. " + "row, " * 3000
    assert len(text) > mining.MAX_CANDIDATE_CHARS
    assert classify_candidate(text) is None


# -- secrets ---------------------------------------------------------------

def test_expanded_secret_families():
    assert is_secret("sk-proj-abcdefghijklmnop1234")
    assert is_secret("sk-ant-api03-abcdefghijklmnop1234")
    assert is_secret("ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert is_secret("github_pat_11ABCDEFG0123456789_abcdefghijklmnop")
    assert is_secret("glpat-abcdefghijklmnopqrst")
    assert is_secret("oc_sk_live_abcdefghijklmnop1234")
    assert is_secret("AKIAIOSFODNN7EXAMPLE")
    assert is_secret("xoxb-123456789012-abcdefghijkl")
    assert is_secret(
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )


def test_prose_credential_assignment_is_secret():
    assert is_secret("update the opencode key with 'oc_sk_abcdefghijklmnop1234'")


# -- role awareness --------------------------------------------------------

def test_assistant_turn_not_stored_as_preference():
    text = "I always prefer to verify the healthcheck before restarting the service"
    assert classify_candidate(text, role="user") == "preference"
    assert classify_candidate(text, role="assistant") == "fact"


def test_fallback_only_applies_to_assistant_turns():
    text = "Your request was not processed. Please try again."
    assert classify_candidate(text, role="assistant") is None
    # Same text from the user is a report of what happened, not boilerplate.
    assert classify_candidate(text, role="user") is not None


# -- drop telemetry (local addition) ---------------------------------------

def test_drops_are_recorded_with_reason():
    mining._DROP_LOG.clear()
    assert classify_candidate("Is plex running without issue?") is None
    drops = recent_drops()
    assert drops and drops[-1]["reason"] == "request"
    assert drops[-1]["chars"] > 0 and drops[-1]["preview"]


def test_drop_log_is_bounded():
    mining._DROP_LOG.clear()
    for _ in range(mining._DROP_LOG.maxlen + 50):
        classify_candidate("no?")
    assert len(mining._DROP_LOG) == mining._DROP_LOG.maxlen
    assert len(recent_drops(limit=5)) == 5


# -- integration through mine_turns ---------------------------------------

def test_mine_turns_filters_and_keeps_role():
    messages = [
        {"role": "user", "content": "Is plex running without issue?"},
        {"role": "assistant", "content": "**Delivered.** message id 5188 uploaded in 46.8s"},
        {"role": "user", "content": "[IMPORTANT: Background process proc_abc completed normally"},
        {
            "role": "user",
            "content": "The vault is growing 12 GB/day, almost entirely application logs.",
        },
    ]
    mined = mine_turns(messages, min_length=20, limit=10)
    assert len(mined) == 1, mined
    assert mined[0]["role"] == "user"
    assert "vault is growing" in mined[0]["content"]

def test_verified_finding_is_not_an_announcement():
    """"Verified." opens findings, not announcements - a "no specifics"
    discriminator was rejected because delivery reports are inherently
    specific (message ids, byte counts, durations)."""
    for text in (
        "Verified. Only settings.json was changed - 19 favorite_models for the opencode provider",
        "Verified - settings.json has 5 available_models, 19 favorite_models, clean bare-model names",
    ):
        assert not looks_like_announcement(text)
        assert classify_candidate(text) is not None


def test_fixed_and_verified_is_still_an_announcement():
    # The phrase form is a completion formula about the work, not an assertion.
    assert looks_like_announcement("**Fixed and verified.** You should have two messages")
    assert classify_candidate("**Fixed and verified.** You should have two messages") is None


def test_announcement_with_specifics_is_still_dropped():
    # Delivery metadata must not rescue an announcement: specificity is what
    # separates facts from non-facts, not what separates findings from reports.
    text = "**Delivered.** The full backup is in this chat - message id 5188, 108 MB, 46.8s"
    assert looks_like_announcement(text)
    assert classify_candidate(text) is None

# -- first-person narration (mined from a real leak row) -------------------

def test_narration_openers_dropped():
    narr = mining.looks_like_narration
    assert narr("Now let me build the full inventory. Writing a scanner that reuses the parser")
    assert narr("Let me verify the store is clean before committing")
    assert narr("Now adding the algorithmic purge pass to the store:")
    assert classify_candidate("Let me verify the store is clean before committing") is None


def test_narration_length_cap_keeps_long_finding():
    text = (
        "Let me be precise: the vault grows 12 GB/day, almost entirely application "
        "logs, so retention rather than capacity is the constraint. " + "Detail. " * 40
    )
    assert len(text) > mining.MAX_NARRATION_CHARS
    assert not mining.looks_like_narration(text)
    assert classify_candidate(text) is not None


def test_narration_has_its_own_drop_reason():
    mining._DROP_LOG.clear()
    assert classify_candidate("Let me check the gateway state") is None
    assert recent_drops()[-1]["reason"] == "narration"

# -- report/payload discriminator + narration lead-in ----------------------

def test_announcement_with_payload_is_kept():
    # A completion formula followed by a finding is knowledge, not noise.
    for text in (
        "**Delivered.** Root cause was NODE_OPTIONS set to 4096 in the unit file.",
        "**Done.** The vault grew because log rotation was disabled in the compose file.",
        "**Fixed and verified.** The lock file is held by the kanban dispatcher.",
    ):
        assert classify_candidate(text, role="assistant") is not None, text


def test_delivery_report_is_still_dropped():
    # Report metadata must not rescue an announcement.
    for text in (
        "**Delivered.** The full backup is in this chat - message id 5188, uploaded in 46.8s",
        "Done. Everything is applied.",
    ):
        assert classify_candidate(text, role="assistant") is None, text


def test_narration_leadin_without_now():
    # Colon-terminated header, and a short clause that asserts nothing.
    for text in (
        "Adding the algorithmic purge pass to the store:",
        "Updating the config to point at the new endpoint.",
        "Writing the migration script against the live store.",
    ):
        assert mining.looks_like_narration(text), text
        assert classify_candidate(text) is None, text


def test_narration_assertions_are_kept():
    for text in (
        "Adding a column means the migration must run first.",
        "Running the retention pass archives rather than deletes, which is why growth looked linear",
    ):
        assert not mining.looks_like_narration(text), text
        assert classify_candidate(text) is not None, text
