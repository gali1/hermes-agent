"""Tests for the Google Custom Search (CSE) provider — the keyed fallback tier.

Covers:
- is_available() gating on BOTH ``GOOGLE_CSE_API_KEY`` and ``GOOGLE_CSE_ID`` (cheap, no network)
- search(): happy path, HTTP error, request error, malformed JSON, non-object/malformed bodies
- result normalisation to the tree's ``{title, url, description, position}`` shape
- resolution order: explicit ``web.backend`` / ``web.search_backend`` always wins; the provider is
  picked when keyed and nothing legacy is available; skipped without a key; it beats the keyless
  tier; it never pre-empts a legacy vendor
- an HTTP failure falls through to the next provider in the order (the tree's one-shot rescue)
- no credential value ever reaches the logs or an error string
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from tests.tools.conftest import register_all_web_providers
from plugins.web._common import BaseWebSearchProvider

SECRET_KEY = "AIzaSySENTINEL-do-not-leak-0123456789"
SECRET_CX = "cse-sentinel-0001"

_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# Every web credential this tree probes — stripped so each test states its own world.
_ENV_KEYS = (
    "GOOGLE_CSE_API_KEY", "GOOGLE_CSE_ID",
    "BRAVE_SEARCH_API_KEY", "SEARXNG_URL", "KEENABLE_API_KEY", "TAVILY_API_KEY", "TAVILY_BASE_URL",
    "EXA_API_KEY", "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "FIRECRAWL_GATEWAY_URL",
    "TOOL_GATEWAY_DOMAIN", "TOOL_GATEWAY_USER_TOKEN", "XAI_API_KEY",
)

_SAMPLE_PAYLOAD = {
    "items": [
        {"title": "A", "link": "https://a.example.com", "snippet": "desc A"},
        {"title": "B", "link": "https://b.example.com", "snippet": "desc B"},
        {"title": "C", "link": "https://c.example.com", "snippet": "desc C"},
    ]
}


def _resp(payload=None, *, status_code: int = 200, json_error: Exception | None = None, text: str = ""):
    """A stand-in ``httpx.Response``; ``status_code >= 400`` makes ``raise_for_status`` raise."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_error is not None:
        resp.json.side_effect = json_error
    else:
        resp.json.return_value = payload
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp,
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def clean_env(monkeypatch):
    """No web credentials from the process env OR ``~/.hermes/.env`` (dev boxes carry real keys)."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: None, raising=True)
    return monkeypatch


@pytest.fixture
def web_registry():
    """All bundled providers registered, registry reset afterwards."""
    register_all_web_providers()
    yield
    from agent.web_search_registry import _reset_for_tests
    _reset_for_tests()


def _google_provider():
    from plugins.web.google_cse.provider import GoogleCSEWebSearchProvider
    return GoogleCSEWebSearchProvider()


def _keyed(monkeypatch):
    """Google CSE keyed: both credentials present."""
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", SECRET_KEY)
    monkeypatch.setenv("GOOGLE_CSE_ID", SECRET_CX)


def _search(limit: int = 5, query: str = "test query"):
    from plugins.web.google_cse import provider as mod
    return mod.GoogleCSEWebSearchProvider().search(query, limit=limit)


class _OtherKeyedProvider(BaseWebSearchProvider):
    """A second *eligible* non-legacy provider, so resolution reaches the walks instead of
    short-circuiting on the single-eligible rule — the tier ordering is then what decides."""

    NAME = "zzz-other"
    DISPLAY_NAME = "Other Keyed"
    KEY_ENV = "ZZZ_OTHER_API_KEY"

    def is_available(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5):
        return {"success": True, "data": {"web": []}}


def _register_other() -> None:
    from agent.web_search_registry import register_provider
    register_provider(_OtherKeyedProvider())


# ---------------------------------------------------------------------------
# Availability — cheap, no network
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_when_both_credentials_set(self, clean_env):
        _keyed(clean_env)
        assert _google_provider().is_available() is True

    def test_unavailable_without_api_key(self, clean_env, caplog):
        clean_env.setenv("GOOGLE_CSE_ID", SECRET_CX)
        with caplog.at_level(logging.DEBUG, logger="plugins.web.google_cse.provider"):
            assert _google_provider().is_available() is False
        assert "GOOGLE_CSE_API_KEY" in caplog.text
        assert "unavailable" in caplog.text
        assert SECRET_CX not in caplog.text

    def test_unavailable_without_engine_id(self, clean_env, caplog):
        """A lone API key cannot address a search engine."""
        clean_env.setenv("GOOGLE_CSE_API_KEY", SECRET_KEY)
        with caplog.at_level(logging.DEBUG, logger="plugins.web.google_cse.provider"):
            assert _google_provider().is_available() is False
        assert "GOOGLE_CSE_ID" in caplog.text
        assert SECRET_KEY not in caplog.text

    def test_is_search_only(self):
        provider = _google_provider()
        assert provider.supports_search() is True
        assert provider.supports_extract() is False
        assert provider.name == "google-cse"

    def test_never_eligible_for_extract(self, clean_env, web_registry):
        """Search-only: the extract capability must never resolve to it."""
        _keyed(clean_env)
        from agent.web_search_registry import get_active_extract_provider, get_active_search_provider
        clean_env.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: False)
        clean_env.setattr("agent.web_search_registry._read_config_key", lambda *path: None)
        assert get_active_search_provider().name == "google-cse"
        assert get_active_extract_provider() is None


# ---------------------------------------------------------------------------
# search() — happy path + normalisation
# ---------------------------------------------------------------------------


class TestSearchNormalisation:
    def test_normalises_to_tree_shape(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp(_SAMPLE_PAYLOAD))

        result = mod.GoogleCSEWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert list(web[0]) == ["title", "url", "description", "position"]
        assert web[0] == {
            "title": "A", "url": "https://a.example.com", "description": "desc A", "position": 1,
        }
        assert web[2]["position"] == 3

    def test_sends_credentials_and_query(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params", {})
            return _resp({"items": []})

        monkeypatch.setattr(mod.httpx, "get", fake_get)
        mod.GoogleCSEWebSearchProvider().search("hello", limit=5)

        assert captured["url"] == _ENDPOINT
        assert captured["params"]["q"] == "hello"
        assert captured["params"]["cx"] == SECRET_CX
        assert captured["params"]["num"] == 5

    def test_limit_capped_at_google_maximum(self, clean_env, monkeypatch):
        """The JSON API serves at most 10 results per request."""
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        captured = {}

        def fake_get(url, **kwargs):
            captured.update(kwargs.get("params", {}))
            return _resp({"items": []})

        monkeypatch.setattr(mod.httpx, "get", fake_get)
        mod.GoogleCSEWebSearchProvider().search("q", limit=50)
        assert captured["num"] == 10

    def test_zero_results_is_success_with_empty_list(self, clean_env, monkeypatch):
        """Google omits ``items`` entirely when nothing matched."""
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp({"searchInformation": {}}))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []


# ---------------------------------------------------------------------------
# search() — degradation, never an exception out of the search path
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_http_error_returns_failure(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp({"error": {}}, status_code=500))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    def test_request_error_returns_failure(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod

        def boom(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(mod.httpx, "get", boom)
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "Could not reach Google CSE" in result["error"]

    def test_malformed_json_returns_failure(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp(json_error=ValueError("not json")))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "Could not parse" in result["error"]

    def test_non_object_body_returns_failure(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp(["unexpected"]))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "unexpected response shape" in result["error"]

    def test_malformed_items_returns_failure(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp({"items": "nope"}))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "unexpected response shape" in result["error"]

    def test_partial_rows_are_skipped_not_fatal(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        payload = {"items": [{"title": "A", "link": "https://a"}, "garbage", {"link": "https://c"}]}
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp(payload))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        assert [row["position"] for row in result["data"]["web"]] == [1, 2]

    def test_error_body_on_http_200_is_a_failure(self, clean_env, monkeypatch):
        """Quota / bad-engine problems arrive as a JSON ``error`` object."""
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        payload = {"error": {"code": 429, "message": "Quota exceeded for quota metric 'Queries'"}}
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp(payload))
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "Quota exceeded" in result["error"]

    def test_search_never_raises(self, clean_env, monkeypatch):
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod

        def boom(url, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(mod.httpx, "get", boom)
        result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
        assert result["success"] is False

    def test_missing_credentials_at_search_time(self, clean_env):
        result = _google_provider().search("q", limit=5)
        assert result["success"] is False
        assert "GOOGLE_CSE_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# Credentials must never leak
# ---------------------------------------------------------------------------


class TestNoCredentialLeak:
    def test_no_credential_in_logs_or_errors(self, clean_env, monkeypatch, caplog):
        """Every failure path, with the vendor echoing the key back at us."""
        _keyed(clean_env)
        from plugins.web.google_cse import provider as mod
        cases = [
            lambda url, **kw: _resp({"error": {"message": f"API key not valid: {SECRET_KEY} for cx {SECRET_CX}"}}),
            lambda url, **kw: _resp({"x": 1}, status_code=403, text=f"key={SECRET_KEY}&cx={SECRET_CX}"),
            lambda url, **kw: _resp(json_error=ValueError(f"bad json for key={SECRET_KEY}")),
        ]
        leaked = []
        with caplog.at_level(logging.DEBUG):
            for stub in cases:
                monkeypatch.setattr(mod.httpx, "get", stub)
                result = mod.GoogleCSEWebSearchProvider().search("q", limit=5)
                assert result["success"] is False
                if SECRET_KEY in json.dumps(result) or SECRET_KEY in caplog.text:
                    leaked.append(result)
        assert not leaked, "credential value reached a log line or an error string"
        assert SECRET_KEY not in caplog.text
        assert SECRET_CX not in caplog.text


# ---------------------------------------------------------------------------
# Resolution order: after the legacy walk, before the keyless tier
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    def _freeze_config(self, monkeypatch, config):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: config)
        monkeypatch.setattr(web_tools, "read_selection", lambda section: None)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        monkeypatch.setattr(
            "agent.web_search_registry._read_config_key",
            lambda *path: (config.get(path[1]) or config.get("backend") or None) if path else None,
        )

    def test_explicit_search_backend_wins(self, clean_env, web_registry, monkeypatch):
        _keyed(clean_env)
        clean_env.setenv("BRAVE_SEARCH_API_KEY", "brave-test")
        self._freeze_config(monkeypatch, {"search_backend": "brave-free"})

        from tools import web_tools
        from agent.web_search_registry import get_active_search_provider
        assert web_tools._get_search_backend() == "brave-free"
        assert get_active_search_provider().name == "brave-free"

    def test_explicit_web_backend_wins(self, clean_env, web_registry, monkeypatch):
        _keyed(clean_env)
        clean_env.setenv("BRAVE_SEARCH_API_KEY", "brave-test")
        self._freeze_config(monkeypatch, {"backend": "brave-free"})

        from tools import web_tools
        from agent.web_search_registry import get_active_search_provider
        assert web_tools._get_backend() == "brave-free"
        assert get_active_search_provider().name == "brave-free"

    def test_selected_when_keyed_and_others_unavailable(self, clean_env, web_registry, monkeypatch):
        _keyed(clean_env)
        self._freeze_config(monkeypatch, {})
        clean_env.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: False)

        from tools import web_tools
        from agent.web_search_registry import get_active_search_provider
        assert web_tools._get_backend() == "google-cse"
        assert web_tools._get_search_backend() == "google-cse"
        assert get_active_search_provider().name == "google-cse"

    def test_skipped_without_key(self, clean_env, web_registry, monkeypatch, caplog):
        """No Google key: the next provider in the order runs instead."""
        clean_env.setenv("BRAVE_SEARCH_API_KEY", "brave-test")
        self._freeze_config(monkeypatch, {})
        _register_other()  # forces the walk past the single-eligible shortcut

        from tools import web_tools
        from agent import web_search_registry as wsr
        assert web_tools._get_search_backend() == "brave-free"
        with caplog.at_level(logging.DEBUG, logger="agent.web_search_registry"):
            assert wsr.get_active_search_provider().name == "brave-free"
            assert wsr._resolve(None, capability="search").name == "brave-free"
        assert "reason: legacy preference" in caplog.text

    def test_no_provider_at_all_when_keyless_disabled(self, clean_env, web_registry, monkeypatch):
        self._freeze_config(monkeypatch, {})
        clean_env.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: False)
        from agent import web_search_registry as wsr
        assert wsr._resolve(None, capability="search") is None
        assert wsr.get_active_search_provider() is None

    def test_never_pre_empts_a_legacy_vendor(self, clean_env, web_registry, monkeypatch):
        _keyed(clean_env)
        clean_env.setenv("BRAVE_SEARCH_API_KEY", "brave-test")
        clean_env.setenv("EXA_API_KEY", "exa-test")
        self._freeze_config(monkeypatch, {})

        from tools import web_tools
        from agent import web_search_registry as wsr
        assert web_tools._get_backend() == "exa"          # legacy walk order beats google-cse
        assert wsr._resolve(None, capability="search").name == "exa"

    def test_sits_above_the_keyless_tier(self, clean_env, web_registry, monkeypatch, caplog):
        """With the keyless ring live, google-cse still wins: it is keyed."""
        _keyed(clean_env)
        self._freeze_config(monkeypatch, {})
        clean_env.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: True)

        # A second *eligible* provider so resolution reaches the walks instead of the
        # single-eligible shortcut — the fallback tier is then the deciding line.
        _register_other()
        from plugins.web.exa.provider import ExaWebSearchProvider
        assert ExaWebSearchProvider().is_keyless_available() is True  # the tier really is live

        from agent import web_search_registry as wsr
        with caplog.at_level(logging.DEBUG, logger="agent.web_search_registry"):
            assert wsr._resolve(None, capability="search").name == "google-cse"
        assert "reason: keyed fallback tier" in caplog.text


# ---------------------------------------------------------------------------
# Dispatch: end-to-end through web_search_tool
# ---------------------------------------------------------------------------


class TestDispatch:
    def _freeze(self, monkeypatch, config=None):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: config or {})
        monkeypatch.setattr(web_tools, "read_selection", lambda section: None)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr("agent.web_search_registry._read_config_key", lambda *path: None)

    def test_web_search_tool_routes_to_google_cse(self, clean_env, web_registry, monkeypatch):
        _keyed(clean_env)
        self._freeze(monkeypatch)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp(_SAMPLE_PAYLOAD))

        from tools import web_tools
        result = json.loads(web_tools.web_search_tool("cse dispatch probe", limit=2))

        assert result["success"] is True
        rows = result["data"]["web"]
        assert rows and list(rows[0]) == ["title", "url", "description", "position"]
        assert rows[0]["url"] == "https://a.example.com"

    def test_http_error_falls_through_to_the_next_provider(self, clean_env, web_registry, monkeypatch):
        """A failed Google call must not surface an exception: the next tier serves the call."""
        _keyed(clean_env)
        self._freeze(monkeypatch)
        from plugins.web.google_cse import provider as mod
        monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: _resp({"error": {}}, status_code=500))

        from plugins.web import keyless_mcp
        monkeypatch.setattr(
            keyless_mcp, "search_with_failover",
            lambda name, query, limit: {"success": True, "data": {"web": [
                {"title": "rescued", "url": "https://r.example.com", "description": "", "position": 1},
            ]}},
        )

        from tools import web_tools
        result = json.loads(web_tools.web_search_tool("cse http error probe", limit=1))

        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://r.example.com"
        assert result["data"]["rescued_from"] == "google-cse"
        assert "HTTP 500" in result["data"]["backend_error"]
