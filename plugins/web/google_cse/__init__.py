"""Google Programmable Search (Custom Search JSON API) — bundled, auto-loaded.

Keyed fallback tier: see ``plugins.web.google_cse.provider`` and
:data:`agent.web_search_registry._FALLBACK_PREFERENCE`.

Credentials: ``GOOGLE_CSE_API_KEY`` + ``GOOGLE_CSE_ID`` (``GOOGLE_CSE_CX`` accepted as an
alias for the engine id; ``GOOGLE_API_KEY`` is not honoured because it usually holds a
Gemini/Vertex key that cannot call this API).

Search-only. Deliberately fails closed: with either credential missing the provider reports
``not configured`` rather than returning an empty success, so a caller can distinguish
"no results" from "no backend".
"""
from __future__ import annotations

from plugins.web.google_cse.provider import GoogleCSEWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(GoogleCSEWebSearchProvider())
