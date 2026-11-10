"""Tool surface for capability delegation.

The primary model keeps the conversation; this tool is how it hands *one* sub-task to a declared
specialist and gets text back.  It is deliberately narrow — one file, one instruction — because a
broad "delegate anything" tool would be a second agent loop, not a sub-call.

Registered under the existing ``delegation`` toolset, which is enabled for the CLI and Telegram
platforms already, so this works everywhere tools do without new configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DELEGATE_CAPABILITY_SCHEMA = {
    "name": "delegate_capability",
    "description": (
        "Hand one capability-specific sub-task to a specialist model and get its text back. "
        "Use it when a file needs a capability your own model lacks — read a scanned PDF, "
        "interpret an image, transcribe audio, review a large diff. The specialist is chosen "
        "from the operator's declared routing pool (capability first, price as tiebreak); "
        "you keep the conversation and present the result. Returns JSON: success, text, "
        "model, provider, modality, bytes_sent, reason, error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local path to the file to hand over (the cached path of an "
                               "attachment, or any file on disk).",
            },
            "instruction": {
                "type": "string",
                "description": "Exactly what the specialist must return — a question about the "
                               "file, or the transformation to apply. No conversation history "
                               "goes with it.",
            },
            "modality": {
                "type": "string",
                "description": "Optional override: image, audio, pdf, text. Inferred from the "
                               "file suffix when omitted.",
            },
        },
        "required": ["path", "instruction"],
    },
}


def _specialist_available() -> bool:
    """True when the operator's pool declares a model besides the conversation's own."""
    try:
        from agent import capability_delegation as cd

        cfg = cd._live_config()
        _cfg_obj, _rc, pool, _src = cd._pool(cfg)
        provider, primary = cd._primary(cfg)
        others = [c for c in pool if (c.provider, c.model) != (provider, primary)]
        return bool(others)
    except Exception:  # noqa: BLE001 - a probe must never break tool loading
        return False


async def _handle_delegate_capability(args: Dict[str, Any], **kw: Any) -> str:
    from agent import capability_delegation as cd

    path = (args.get("path") or "").strip()
    instruction = (args.get("instruction") or "").strip()
    modality = (args.get("modality") or "").strip()
    if not path or not instruction:
        return tool_error("delegate_capability needs both `path` and `instruction`")

    result = await cd.delegate_capability(path, instruction, modality=modality)
    logger.info(
        "delegate_capability: modality=%s specialist=%s/%s success=%s bytes=%s",
        result.modality, result.provider, result.model, result.success, result.bytes_sent,
    )
    # A failed delegation is a result the primary model must see, not an exception it cannot read.
    return result.to_json()


registry.register(
    name="delegate_capability",
    toolset="delegation",
    schema=DELEGATE_CAPABILITY_SCHEMA,
    handler=_handle_delegate_capability,
    check_fn=_specialist_available,
    is_async=True,
)
