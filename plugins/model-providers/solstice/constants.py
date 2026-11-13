"""HTTP-free constants shared by the Solstice profile and its transport.

Kept out of ``transport.py`` on purpose: ``transport`` imports the native Gemini adapter, which
imports ``httpx`` at module load. Provider discovery imports the plugin package to register it, and
that happens in processes where the dependency environment may not be selected yet (early boot, the
update tool-install phase), so registration must not drag ``httpx`` in. The endpoint lives here; the
client is imported lazily by ``SolsticeProfile.create_client``, where the runtime deps are active.
"""

from __future__ import annotations

INFERENCE_BASE_URL = "https://generativelanguage.googleapis.com/v1alpha"
