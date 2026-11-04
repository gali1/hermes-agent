"""Google Programmable Search (Custom Search JSON API) — search only, keyed fallback tier.

Env: ``GOOGLE_CSE_API_KEY`` (API key) **and** ``GOOGLE_CSE_ID`` (the engine's ``cx`` id).
Config: ``web.search_backend`` / ``web.backend: "google-cse"`` to pin it directly.

Tier: registered in :data:`agent.web_search_registry._FALLBACK_PREFERENCE`, i.e. tried
only AFTER the legacy vendor walk and BEFORE the keyless free tier — it never pre-empts a
legacy vendor and never overrides an explicit ``web.backend`` / ``web.search_backend``.

No credential value is ever logged or returned: the key travels in the query string (the
API's documented form), so every log line and error string is passed through :func:`_redact`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from plugins.web._common import (
    BaseWebSearchProvider, provider_env, run_search, search_fail, search_ok, setup_schema, title_hit,
)

logger = logging.getLogger(__name__)

_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
# The JSON API serves at most 10 results per request (``num``); more needs pagination via ``start``.
_MAX_RESULTS_PER_REQUEST = 10


def _redact(text: str, *secrets: str) -> str:
    """Strip credential values out of *text* before it is logged or returned."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class GoogleCSEWebSearchProvider(BaseWebSearchProvider):
    """Search-only Google Programmable Search provider (keyed fallback tier)."""

    NAME = "google-cse"
    DISPLAY_NAME = "Google Custom Search"
    KEY_ENV = "GOOGLE_CSE_API_KEY"
    CX_ENV = "GOOGLE_CSE_ID"
    EXTRACT = False
    KEYLESS = False

    def is_available(self) -> bool:
        """Both credentials must be present — a lone API key cannot address an engine."""
        for env_var in (self.KEY_ENV, self.CX_ENV):
            if not provider_env(env_var):
                logger.debug("%s unavailable: %s is not set", self.NAME, env_var)
                return False
        return True

    def _search_body(self, query: str, limit: int) -> Dict[str, Any]:
        api_key = provider_env(self.KEY_ENV)
        cx = provider_env(self.CX_ENV)
        if not api_key or not cx:
            return search_fail(f"{self.KEY_ENV} and {self.CX_ENV} must both be set for Google Custom Search")

        num = max(1, min(int(limit), _MAX_RESULTS_PER_REQUEST))
        params = {"key": api_key, "cx": cx, "q": query, "num": num}
        try:
            resp = httpx.get(_ENDPOINT, params=params, headers={"Accept": "application/json"}, timeout=15)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Google CSE HTTP error: %s", _redact(str(exc), api_key, cx))
            return search_fail(f"Google CSE returned HTTP {exc.response.status_code}")
        except httpx.RequestError as exc:
            detail = _redact(str(exc), api_key, cx)
            logger.warning("Google CSE request error: %s", detail)
            return search_fail(f"Could not reach Google CSE: {detail}")

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — malformed body is a failure, never an exception
            logger.warning("Google CSE response parse error: %s", _redact(str(exc), api_key, cx))
            return search_fail("Could not parse Google CSE response as JSON")

        if not isinstance(data, dict):
            logger.warning(
                "Google CSE returned a non-object response (%s)", type(data).__name__,
            )
            return search_fail("Google CSE returned an unexpected response shape")

        # Quota / credential / bad-engine problems arrive as a JSON ``error`` object; the status
        # code is not always >=400, so the body is checked as well.
        error = data.get("error")
        if isinstance(error, dict):
            message = _redact(str(error.get("message") or "unknown error"), api_key, cx)
            logger.warning("Google CSE error response: %s", message)
            return search_fail(f"Google CSE error: {message}")

        items = data.get("items") or []
        if not isinstance(items, list):
            return search_fail("Google CSE returned an unexpected response shape")

        raw = [item for item in items[:num] if isinstance(item, dict)]
        rows = [
            title_hit(str(item.get("title", "")), str(item.get("link", "")), str(item.get("snippet", "")), i + 1)
            for i, item in enumerate(raw)
        ]
        logger.info("Google CSE '%s': %d results (from %d raw, limit %d)", query, len(rows), len(items), limit)
        return search_ok(rows)

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        return run_search("Google CSE", logger, lambda: self._search_body(query, limit))

    def get_setup_schema(self) -> Dict[str, Any]:
        schema = setup_schema(
            self.DISPLAY_NAME, "free",
            "Keyed fallback — needs GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID (100 queries/day free). Search only.",
            self.KEY_ENV, "Google API key (Custom Search JSON API)",
            "https://developers.google.com/custom-search/v1/overview",
        )
        schema["env_vars"].append({
            "key": self.CX_ENV,
            "prompt": "Search engine ID (cx)",
            "url": "https://programmablesearchengine.google.com/controlpanel/all",
        })
        return schema
