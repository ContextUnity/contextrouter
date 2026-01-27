"""ContextUnit -> AG-UI mapping helpers.

Transforms ContextUnit payloads into frontend-friendly AG-UI event dicts.
This is intentionally minimal and additive (does not change existing SSE shapes).
"""

from __future__ import annotations

from typing import Any

from contextcore import ContextUnit

from contextrouter.modules.retrieval.rag.formatting.citations import format_citations_to_ui


def context_unit_to_agui_event(unit: ContextUnit) -> dict[str, Any]:
    """Convert ContextUnit into a generic AG-UI event payload."""

    payload = unit.payload or {}
    token_id = payload.get("token_id") if isinstance(payload, dict) else None

    return {
        "type": "ContextUnit",
        "tokenId": token_id,
        "provenance": list(unit.provenance or []),
        "citations": format_citations_to_ui(
            payload.get("citations", []) if isinstance(payload, dict) else []
        ),
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
        "data": payload.get("data") or payload.get("content"),
    }


__all__ = ["context_unit_to_agui_event"]
