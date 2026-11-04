"""T1f — cross-provider selection at the turn boundary.

Two behaviours are pinned here:

(a) an enabled router that selects a model living on ANOTHER *configured*
    provider must apply the switch, with credentials resolved through that
    provider's saved profile, so the rebuilt client talks to the new endpoint
    with that provider's key — not the previous provider's;
(b) a provider that has no usable profile must be declined at the boundary
    with a log line that says why (naming the credential *variable*, never its
    value), instead of half-switching the agent or building a request that can
    only 401.

No network: ``resolve_runtime_provider`` is pure config+env lookup, and the
provider config it reads is supplied by the ``profile`` fixture below.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.models_dev import ModelInfo  # noqa: E402
from agent import model_router as mr  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _info(model, *, provider, ctx=200_000, cost_in=1.0, cost_out=2.0):
    return ModelInfo(
        id=model,
        name=model,
        family="",
        provider_id=provider,
        reasoning=False,
        tool_call=True,
        attachment=False,
        structured_output=False,
        input_modalities=("text",),
        output_modalities=("text",),
        context_window=ctx,
        max_output=16_384,
        cost_input=cost_in,
        cost_output=cost_out,
        status="",
    )


#: Deliberately priced so the cross-provider candidate wins on merit: the
#: session provider's model is expensive, the two foreign ones are cheap.
_SPECS = {
    ("sess", "session-model"): dict(cost_in=50.0, cost_out=50.0),
    ("prov-b", "model-b"): dict(cost_in=0.1, cost_out=0.2),
    ("prov-c", "model-c"): dict(cost_in=0.1, cost_out=0.2),
}


class _StubAgent:
    """Mirrors ``agent_runtime_helpers.switch_model`` closely enough to observe
    what the *next* request would be built from: provider, endpoint, key, mode.
    """

    def __init__(self, model="session-model", provider="sess"):
        self.model = model
        self.provider = provider
        self.api_mode = "chat_completions"
        self.api_key = "key-of-the-old-provider"
        self.base_url = "http://old.example/v1"
        self.tools = None
        self.switch_calls = []

    def switch_model(
        self, new_model, new_provider, api_key="", base_url="", api_mode="", user_initiated=True
    ):
        self.switch_calls.append(
            {
                "model": new_model,
                "provider": new_provider,
                "api_key": api_key,
                "base_url": base_url,
                "api_mode": api_mode,
                "user_initiated": user_initiated,
            }
        )
        self.model = new_model
        self.provider = new_provider
        # What the real helper does: rebuild the client from these values.
        self.api_key = api_key or self.api_key
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode or self.api_mode


def _config(*, candidates, extra_providers=None, default="session-model"):
    providers = {"sess": {"models": ["session-model"]}}
    providers.update(extra_providers or {})
    return {
        "smart_model_routing": {"enabled": True, "candidates": list(candidates)},
        "providers": providers,
        "model": {"provider": "sess", "default": default},
    }


@pytest.fixture
def profile(monkeypatch):
    """Make ``resolve_runtime_provider`` read the same providers: block the test
    builds.  In production both call sites read one ``config.yaml``; here the
    in-memory config is the file.
    """
    import hermes_cli.runtime_provider as rp

    cfg = {}

    def _load_config():
        return cfg

    monkeypatch.setattr(rp, "load_config", _load_config, raising=False)
    # Credential pools are a separate on-disk store: keep them out of this test.
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None, raising=False)
    return cfg


@pytest.fixture
def catalog(monkeypatch):
    """Deterministic, offline model catalog (models.dev is never consulted)."""
    calls = []

    def _candidates_for(provider, models, *, enabled=True):
        calls.append((provider, list(models)))
        return [
            mr.ModelCandidate(
                provider=provider,
                model=m,
                info=_info(m, provider=provider, **{k: v for k, v in _SPECS[(provider, m)].items()}),
                enabled=enabled,
            )
            for m in models
            if (provider, m) in _SPECS
        ]

    monkeypatch.setattr(mr, "candidates_for", _candidates_for)
    return calls


# ---------------------------------------------------------------------------
# (a) configured provider B: switch applied, credentials from B's profile
# ---------------------------------------------------------------------------

def test_cross_provider_switch_uses_the_target_providers_credentials(
    profile, catalog, monkeypatch, caplog
):
    monkeypatch.setenv("PROV_B_API_KEY", "sk-b-secret-value")
    monkeypatch.delenv("PROV_C_API_KEY", raising=False)

    cfg = _config(
        candidates=["prov-b/model-b"],
        extra_providers={
            "prov-b": {
                "name": "Provider B",
                "base_url": "http://b.example/v1",
                "key_env": "PROV_B_API_KEY",
                "models": ["model-b"],
            }
        },
    )
    profile.update(cfg)

    agent = _StubAgent()
    with caplog.at_level(logging.INFO, logger="hermes.model_router"):
        decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)

    assert decision is not None, "a configured cross-provider model must be usable"
    assert (decision.provider, decision.model) == ("prov-b", "model-b")
    assert decision.switched is True

    # The swap reached the agent, and it carried B's profile with it.
    assert len(agent.switch_calls) == 1
    call = agent.switch_calls[0]
    assert (call["provider"], call["model"]) == ("prov-b", "model-b")
    assert call["user_initiated"] is False, "an automatic switch must not pin the session"
    assert call["base_url"] == "http://b.example/v1", "request must be rebuilt on B's endpoint"
    assert call["api_key"] == "sk-b-secret-value", "key must come from B's key_env variable"
    assert call["api_mode"] == "chat_completions"

    # ...and the agent's live request state moved with it, so the next API call
    # does not still target the old provider with the old key.
    assert agent.base_url == "http://b.example/v1"
    assert agent.api_key == "sk-b-secret-value"
    assert agent.provider == "prov-b"

    # Logs may name the variable; they may never carry the value.
    assert "sk-b-secret-value" not in caplog.text


# ---------------------------------------------------------------------------
# (b) provider C declares an endpoint but has no credentials: skip, and say why
# ---------------------------------------------------------------------------

def test_unconfigured_provider_is_skipped_with_the_reason_logged(
    profile, catalog, monkeypatch, caplog
):
    monkeypatch.delenv("PROV_C_API_KEY", raising=False)

    cfg = _config(
        candidates=["prov-c/model-c"],
        extra_providers={
            "prov-c": {
                "name": "Provider C",
                "base_url": "http://c.example/v1",
                "key_env": "PROV_C_API_KEY",
                "models": ["model-c"],
            }
        },
    )
    profile.update(cfg)

    agent = _StubAgent()
    with caplog.at_level(logging.INFO, logger="hermes.model_router"):
        decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)

    # The turn is declined, not half-applied.
    assert decision is None
    assert agent.switch_calls == []
    assert agent.model == "session-model"
    assert agent.provider == "sess"
    assert agent.base_url == "http://old.example/v1"
    assert agent.api_key == "key-of-the-old-provider"

    # ...and the recorded reason names the model that was declined, the
    # provider, and the credential variable that is missing.
    reasons = [
        r.getMessage() for r in caplog.records
        if r.name == "hermes.model_router" and "skipped" in r.getMessage()
    ]
    assert reasons, f"expected a skip reason; got {[r.getMessage() for r in caplog.records]}"
    reason = reasons[-1]
    assert "prov-c/model-c" in reason
    assert "prov-c" in reason
    assert "PROV_C_API_KEY" in reason


# ---------------------------------------------------------------------------
# The two edges the boundary must not cross
# ---------------------------------------------------------------------------

def test_unknown_provider_is_never_handed_to_the_aggregator(monkeypatch):
    """``resolve_runtime_provider`` falls through to OpenRouter for names it
    does not know; routing must decline before that, not silently re-route the
    model to an aggregator the operator never assigned it to."""
    import hermes_cli.runtime_provider as rp

    def _boom(**_kwargs):
        raise AssertionError("an unconfigured provider must not be resolved at all")

    monkeypatch.setattr(rp, "resolve_runtime_provider", _boom)

    runtime, reason = mr._resolve_provider_runtime({}, "acme-relay", "some-model")

    assert runtime is None
    assert "acme-relay" in reason
    assert "not configured" in reason


def test_local_profile_without_a_key_is_still_usable(profile, catalog, monkeypatch):
    """A profile that declares no credential (a local server) must not be
    mistaken for an unconfigured one."""
    cfg = _config(
        candidates=["local/model-c"],
        extra_providers={
            "local": {"name": "Local", "base_url": "http://127.0.0.1:11434/v1", "models": ["model-c"]}
        },
    )
    profile.update(cfg)
    # Give the local provider its own catalog entry under the right name.
    monkeypatch.setitem(_SPECS, ("local", "model-c"), dict(cost_in=0.1, cost_out=0.2))

    agent = _StubAgent()
    decision = mr.route_turn(agent, "What is 15 multiplied by 8?", config=cfg)

    assert decision is not None
    assert (decision.provider, decision.model) == ("local", "model-c")
    assert agent.base_url == "http://127.0.0.1:11434/v1"
    assert agent.provider == "local"

