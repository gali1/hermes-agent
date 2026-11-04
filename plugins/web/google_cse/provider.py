"""Google Custom Search JSON API (``https://www.googleapis.com/customsearch/v1``).

Search-only — CSE returns results but does not fetch page bodies.
Env: ``GOOGLE_CSE_API_KEY`` and ``GOOGLE_CSE_CX`` (both required). ``GOOGLE_API_KEY`` and
``GOOGLE_CSE_ID`` are accepted as aliases so an existing Google setup keeps working; the
canonical pair is the one both implementations document and the one the setup prompt names.

Design notes:
* **Fails closed.** Missing credentials return ``not configured``, never an empty success —
  a caller must be able to tell "the backend is absent" from "the query found nothing".
* **Never leaks the key.** The API key travels as a query parameter to Google and is never
  included in an error message, a log line or the returned payload. Presence is checked by
  variable *name* only; no value is ever logged.
* **Bounded.** CSE accepts ``num`` up to 10; larger limits are clamped rather than paginated,
  because a fallback backend should not multiply billable calls. One request per query — a
  transport or parse failure surfaces as provider-unavailable so the next provider can run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from plugins.web._common import (
    BaseWebSearchProvider,
    http_get_json,
    provider_env,
    search_fail,
    search_ok,
    setup_schema,
    titled_rows,
)

logger = logging.getLogger(__name__)

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
_MAX_NUM = 10  # hard API ceiling per request


class GoogleCSEWebSearchProvider(BaseWebSearchProvider):
    """Search via Google's Custom Search JSON API."""

    NAME = "google_cse"
    DISPLAY_NAME = "Google Custom Search"
    KEY_ENV = "GOOGLE_CSE_API_KEY"
    SEARCH_ID_ENV = "GOOGLE_CSE_CX"
    # Legacy spellings, accepted after the canonical names. Order matters: the canonical
    # variable wins when both are set, so one box can migrate without editing the other.
    KEY_ENV_ALIASES: Tuple[str, ...] = ("GOOGLE_API_KEY",)
    SEARCH_ID_ENV_ALIASES: Tuple[str, ...] = ("GOOGLE_CSE_ID",)

    @staticmethod
    def _resolve_env(canonical: str, aliases: Tuple[str, ...]) -> Tuple[str, str]:
        """First set variable's value and the name it came from; ``("", canonical)`` if none."""
        for name in (canonical, *aliases):
            value = provider_env(name)
            if value:
                return value, name
        return "", canonical

    def _credentials(self) -> Tuple[str, str, list]:
        api_key, _key_name = self._resolve_env(self.KEY_ENV, self.KEY_ENV_ALIASES)
        cse_id, _id_name = self._resolve_env(self.SEARCH_ID_ENV, self.SEARCH_ID_ENV_ALIASES)
        missing = [name for name, value in ((self.KEY_ENV, api_key), (self.SEARCH_ID_ENV, cse_id))
                   if not value]
        return api_key, cse_id, missing

    def is_available(self) -> bool:
        """Both halves of the credential pair are required; one alone cannot search."""
        api_key, cse_id, _missing = self._credentials()
        return bool(api_key and cse_id)

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        api_key, cse_id, missing = self._credentials()
        if missing:
            return search_fail(
                f"Google Custom Search not configured: {', '.join(missing)} unset")
        num = max(1, min(int(limit or 5), _MAX_NUM))
        data, failure = http_get_json(
            "Google Custom Search",
            GOOGLE_CSE_ENDPOINT,
            params={"key": api_key, "cx": cse_id, "q": query, "num": num},
            headers={"Accept": "application/json"},
            timeout=15,
            logger=logger,
            reach_target="Google Custom Search API",
        )
        if failure is not None:
            return failure
        items = (data or {}).get("items") or []
        rows = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("link", "")),
                "content": str(item.get("snippet", "")),
            }
            for item in items[:num]
            if isinstance(item, dict)
        ]
        web_results = titled_rows(rows, "content")
        logger.info(
            "Google CSE search '%s': %d results (limit %d)", query, len(web_results), num)
        return search_ok(web_results)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "Google Custom Search",
            "paid · 100 queries/day free",
            "Google Programmable Search. Needs both GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX "
            "(GOOGLE_API_KEY / GOOGLE_CSE_ID are accepted aliases).",
            "GOOGLE_CSE_API_KEY",
            "Google API key (and GOOGLE_CSE_CX for the search engine)",
            "https://developers.google.com/custom-search/v1/overview",
        )
