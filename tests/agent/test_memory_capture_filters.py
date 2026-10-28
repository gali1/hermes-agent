"""Capture filters for the two classes heuristics used to store: requests and
assistant status lines.

Ground truth for the junk cases is real rows that reached the live store via
auto-capture (an interrogative, a meta-command, and a canned failure line);
ground truth for the signal cases is real root-cause notes that must survive.
"""

from agent.memory.mining import (
    classify_candidate,
    is_request,
    looks_injected,
    looks_like_announcement,
    looks_like_fallback,
    mine_turns,
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
    "Ok, add the retention note to the skill and commit.",
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


def test_announcement_needs_no_specifics():
    """A completion verb alone is not enough; naming an artifact makes it a finding."""
    assert looks_like_announcement("Delivered. message id 5188")
    assert looks_like_announcement("All done — shipped.")
    assert looks_like_announcement("Done.")
    assert not looks_like_announcement(
        "Verified. Only `settings.json` was changed — 19 `favorite_models` for the opencode provider."
    )
    assert not looks_like_announcement(
        "Verified — settings.json has 5 available_models and clean bare-model names everywhere"
    )
    assert not looks_like_announcement("Fixed and verified the deploy, port 8080 answers again")

    kept = "Verified. Only `settings.json` was changed — 19 `favorite_models` for the opencode provider."
    assert classify_candidate(kept, role="assistant") is not None
    assert classify_candidate("Delivered. message id 5188", role="assistant") is None


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
