"""Google Custom Search provider — fail-closed behaviour, mapping, and key hygiene."""

from __future__ import annotations

import pytest

from plugins.web.google_cse import provider as gcse


@pytest.fixture
def env(monkeypatch):
    """Controllable provider_env; the real one reads os.environ then ~/.hermes/.env."""
    values: dict[str, str] = {}
    monkeypatch.setattr(gcse, "provider_env", lambda name: values.get(name, ""))
    return values


class TestAvailability:
    def test_unavailable_without_credentials(self, env):
        p = gcse.GoogleCSEWebSearchProvider()
        assert p.is_available() is False

    def test_unavailable_with_only_api_key(self, env):
        env["GOOGLE_CSE_API_KEY"] = "k"
        assert gcse.GoogleCSEWebSearchProvider().is_available() is False

    def test_unavailable_with_only_cse_id(self, env):
        env["GOOGLE_CSE_CX"] = "c"
        assert gcse.GoogleCSEWebSearchProvider().is_available() is False

    def test_available_with_both_canonical_names(self, env):
        env["GOOGLE_CSE_API_KEY"] = "k"
        env["GOOGLE_CSE_CX"] = "c"
        assert gcse.GoogleCSEWebSearchProvider().is_available() is True

    def test_legacy_aliases_still_work(self, env):
        """An install configured before the rename keeps working unchanged."""
        env["GOOGLE_API_KEY"] = "k"
        env["GOOGLE_CSE_ID"] = "c"
        assert gcse.GoogleCSEWebSearchProvider().is_available() is True

    def test_canonical_name_wins_when_both_are_set(self, env, monkeypatch):
        env["GOOGLE_CSE_API_KEY"] = "canonical-key"
        env["GOOGLE_API_KEY"] = "legacy-key"
        env["GOOGLE_CSE_CX"] = "canonical-cx"
        env["GOOGLE_CSE_ID"] = "legacy-cx"
        captured = {}

        def fake_get(label, url, *, params, headers, timeout, logger, reach_target=None):
            captured.update(params)
            return {"items": []}, None

        monkeypatch.setattr(gcse, "http_get_json", fake_get)
        gcse.GoogleCSEWebSearchProvider().search("q")
        assert captured["key"] == "canonical-key" and captured["cx"] == "canonical-cx"


class TestSearch:
    def test_fails_closed_and_names_the_missing_variable(self, env):
        env["GOOGLE_CSE_API_KEY"] = "k"          # CSE id deliberately absent
        result = gcse.GoogleCSEWebSearchProvider().search("anything")
        assert result["success"] is False
        assert "GOOGLE_CSE_CX" in result["error"]
        assert "not configured" in result["error"]

    def test_success_maps_items_to_rows(self, env, monkeypatch):
        env["GOOGLE_CSE_API_KEY"] = "k"
        env["GOOGLE_CSE_CX"] = "c"
        captured = {}

        def fake_get(label, url, *, params, headers, timeout, logger, reach_target=None):
            captured.update(params)
            return {"items": [
                {"title": "One", "link": "https://a.example", "snippet": "first"},
                {"title": "Two", "link": "https://b.example", "snippet": "second"},
            ]}, None

        monkeypatch.setattr(gcse, "http_get_json", fake_get)
        result = gcse.GoogleCSEWebSearchProvider().search("q", limit=2)
        assert result["success"] is True
        rows = result["data"]["web"]
        assert [r["url"] for r in rows] == ["https://a.example", "https://b.example"]
        assert captured["num"] == 2 and captured["cx"] == "c"

    def test_limit_is_clamped_to_api_ceiling(self, env, monkeypatch):
        env["GOOGLE_CSE_API_KEY"] = "k"
        env["GOOGLE_CSE_CX"] = "c"
        captured = {}

        def fake_get(label, url, *, params, headers, timeout, logger, reach_target=None):
            captured.update(params)
            return {"items": []}, None

        monkeypatch.setattr(gcse, "http_get_json", fake_get)
        gcse.GoogleCSEWebSearchProvider().search("q", limit=50)
        assert captured["num"] == 10

    def test_no_items_is_a_successful_empty_result(self, env, monkeypatch):
        env["GOOGLE_CSE_API_KEY"] = "k"
        env["GOOGLE_CSE_CX"] = "c"
        monkeypatch.setattr(gcse, "http_get_json",
                            lambda *a, **k: ({}, None))
        result = gcse.GoogleCSEWebSearchProvider().search("q")
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_transport_failure_is_propagated_without_the_key(self, env, monkeypatch):
        env["GOOGLE_CSE_API_KEY"] = "SECRET_KEY_VALUE"
        env["GOOGLE_CSE_CX"] = "c"
        monkeypatch.setattr(gcse, "http_get_json",
                            lambda *a, **k: (None, {"success": False, "error": "HTTP 403"}))
        result = gcse.GoogleCSEWebSearchProvider().search("q")
        assert result["success"] is False
        assert "SECRET_KEY_VALUE" not in result["error"]

    def test_one_request_per_query(self, env, monkeypatch):
        """A fallback backend must not multiply billable calls."""
        env["GOOGLE_CSE_API_KEY"] = "k"
        env["GOOGLE_CSE_CX"] = "c"
        calls = []

        def fake_get(*a, **k):
            calls.append(1)
            return {"items": []}, None

        monkeypatch.setattr(gcse, "http_get_json", fake_get)
        gcse.GoogleCSEWebSearchProvider().search("q", limit=10)
        assert len(calls) == 1


class _FakeProvider:
    """Minimal WebSearchProvider stand-in for registry ordering assertions."""

    def __init__(self, name, *, available=True, keyless=False, search=True):
        self.name = name
        self._available = available
        self._keyless = keyless
        self._search = search

    def supports_search(self):
        return self._search

    def supports_extract(self):
        return False

    def is_available(self):
        return self._available

    def is_keyless_available(self):
        return self._keyless


class TestRegistryOrdering:
    """google_cse sits after the legacy walk and before the keyless tier."""

    def _install(self, monkeypatch, providers, *, keyless=("exa",)):
        from agent import web_search_registry as reg

        snapshot = {p.name: p for p in providers}
        monkeypatch.setattr(reg._registry, "merged", lambda: dict(snapshot))
        monkeypatch.setattr(reg, "_keyless_preference", lambda: tuple(keyless))
        monkeypatch.setattr(reg, "_keyless_tier_enabled", lambda: True)
        return reg

    @staticmethod
    def _name(reg, configured=None):
        provider = reg._resolve(configured, capability="search")
        assert provider is not None
        return provider.name

    def test_legacy_backend_is_not_downgraded(self, monkeypatch):
        reg = self._install(monkeypatch, [
            _FakeProvider("tavily"), _FakeProvider("google_cse"), _FakeProvider("exa", available=False, keyless=True)])
        assert self._name(reg) == "tavily"

    def test_google_cse_beats_the_keyless_tier(self, monkeypatch):
        reg = self._install(monkeypatch, [
            _FakeProvider("google_cse"), _FakeProvider("exa", available=False, keyless=True)])
        assert self._name(reg) == "google_cse"

    def test_google_cse_is_reached_when_it_is_the_only_keyed_backend(self, monkeypatch):
        reg = self._install(monkeypatch, [
            _FakeProvider("google_cse"), _FakeProvider("exa", available=False, keyless=True)])
        assert self._name(reg) == "google_cse"

    def test_explicit_backend_still_wins(self, monkeypatch):
        reg = self._install(monkeypatch, [
            _FakeProvider("google_cse"), _FakeProvider("tavily")])
        assert self._name(reg, "google_cse") == "google_cse"
        assert self._name(reg, "tavily") == "tavily"

    def test_unavailable_google_cse_falls_through(self, monkeypatch):
        reg = self._install(monkeypatch, [
            _FakeProvider("google_cse", available=False),
            _FakeProvider("exa", available=False, keyless=True)])
        assert self._name(reg) == "exa"

    def test_choice_and_reason_are_logged_at_debug(self, monkeypatch, caplog):
        import logging

        reg = self._install(monkeypatch, [
            _FakeProvider("google_cse"), _FakeProvider("exa", available=False, keyless=True)])
        with caplog.at_level(logging.DEBUG, logger="agent.web_search_registry"):
            reg._resolve(None, capability="search")
        assert "keyed tier after legacy walk" in caplog.text
        assert "google_cse" in caplog.text
