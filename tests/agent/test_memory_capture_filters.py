"""Capture filters for turn mining: requests, status lines, injected prompts.

Two sources feed these cases.  The request/fallback/injected/secret classes come
from the dev-box implementation of ``agent/memory/mining.py``; the brief,
announcement and drop-telemetry cases are local additions made after a shadow
run over this host's live store showed the 3 KB cap clipping genuine findings
while a "**Delivered.** …" status line survived as a fact.

Every filter here is pure regex - no model call, no tokens.
"""

import agent.memory.mining as mining
from agent.memory.mining import (
    classify_candidate,
    is_request,
    is_secret,
    looks_like_announcement,
    looks_like_brief,
    looks_like_fallback,
    looks_injected,
    mine_turns,
    recent_drops,
)


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
