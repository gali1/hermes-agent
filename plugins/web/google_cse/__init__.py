"""Google Custom Search (CSE) plugin — bundled, auto-loaded.

Keyed fallback tier: see ``plugins.web.google_cse.provider``.
"""
from __future__ import annotations
from plugins.web.google_cse.provider import GoogleCSEWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(GoogleCSEWebSearchProvider())
