"""ContextUnit -> AG-UI mapping helpers.

Transforms ContextUnit payloads into frontend-friendly AG-UI event dicts.
This is intentionally minimal and additive (does not change existing SSE shapes).
"""

from __future__ import annotations

from contextunity.core import ContextUnit
from contextunity.core.types import (
    ContextUnitPayload,
    JsonDict,
    JsonValue,
    is_json_dict,
    is_json_value,
    is_object_list,
)

from contextunity.router.modules.retrieval.rag.formatting.citations import format_citations_to_ui
from contextunity.router.modules.retrieval.rag.models import Citation
from contextunity.router.modules.retrieval.rag.types import UICitation

from .events import AguiEventDict


def _citations_from_payload(payload: ContextUnitPayload) -> list[Citation]:
    """Parse wire payload citations into validated ``Citation`` models."""
    raw = payload.get("citations")
    if not is_object_list(raw):
        return []
    citations: list[Citation] = []
    for item in raw:
        if isinstance(item, Citation):
            citations.append(item)
        elif isinstance(item, dict):
            try:
                citations.append(Citation.model_validate(item))
            except Exception:
                continue
    return citations


def _ui_citations_as_json(citations: list[UICitation]) -> list[JsonDict]:
    """Narrow formatted UI citations to JSON-safe dict rows."""
    out: list[JsonDict] = []
    for citation in citations:
        raw = dict(citation)
        if is_json_dict(raw):
            out.append(raw)
    return out


def _payload_data(payload: ContextUnitPayload) -> JsonValue | None:
    """Extract JSON-safe event data from a ContextUnit payload."""
    data_raw = payload.get("data")
    if is_json_value(data_raw):
        return data_raw
    content_raw = payload.get("content")
    if is_json_value(content_raw):
        return content_raw
    return None


def contextunit_to_agui_event(unit: ContextUnit) -> AguiEventDict:
    """Convert ContextUnit into a generic AG-UI event payload."""
    payload: ContextUnitPayload = unit.payload or {}
    token_id = payload.get("token_id")
    metadata_raw = payload.get("metadata")
    metadata = metadata_raw if is_json_dict(metadata_raw) else {}

    ui_citations = format_citations_to_ui(_citations_from_payload(payload))

    return AguiEventDict(
        type="ContextUnit",
        tokenId=token_id if isinstance(token_id, str) else None,
        provenance=list(unit.provenance or []),
        citations=_ui_citations_as_json(ui_citations),
        metadata=metadata,
        data=_payload_data(payload),
    )


__all__ = ["contextunit_to_agui_event"]
