"""Parse ``brain_event`` custom callback payloads into typed fields."""

from __future__ import annotations

from contextunity.core.types import is_object_dict

from contextunity.router.cortex.events import BrainEvent
from contextunity.router.modules.observability.contracts import span_children


def _event_data_dict(data: object) -> dict[str, object]:
    if is_object_dict(data):
        return dict(data)
    return {}


def parse_brain_custom_event(
    data: object,
) -> tuple[str, str | None, dict[str, object]] | None:
    """Return ``(event_type, node_name, payload)`` from a custom callback payload."""
    if not is_object_dict(data):
        return None

    event_raw = data.get("event")
    if isinstance(event_raw, BrainEvent):
        return event_raw.type, event_raw.node, _event_data_dict(event_raw.data)

    if not is_object_dict(event_raw):
        return None

    type_raw = event_raw.get("type")
    if not isinstance(type_raw, str):
        return None

    node_raw = event_raw.get("node")
    node_name = node_raw if isinstance(node_raw, str) else None
    payload = _event_data_dict(event_raw.get("data"))
    return type_raw, node_name, payload


__all__ = ["parse_brain_custom_event", "span_children"]
