"""Bisquit -> AG-UI mapping helpers.

Transforms BisquitEnvelope payloads into frontend-friendly AG-UI event dicts.
This is intentionally minimal and additive (does not change existing SSE shapes).
"""

from __future__ import annotations

from typing import Any

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.modules.retrieval.rag.formatting.citations import format_citations_to_ui


def bisquit_envelope_to_agui_event(envelope: BisquitEnvelope) -> dict[str, Any]:
    """Convert BisquitEnvelope into a generic AG-UI event payload."""

    return {
        "type": "BisquitEnvelope",
        "tokenId": envelope.token_id,
        "provenance": list(envelope.provenance or []),
        "citations": format_citations_to_ui(envelope.citations or []),
        "metadata": dict(envelope.metadata or {}),
        "data": envelope.data,
    }


__all__ = ["bisquit_envelope_to_agui_event"]
