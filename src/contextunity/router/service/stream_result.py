"""Validation helpers for federated stream executor payloads."""

from __future__ import annotations

from contextunity.core.types import JsonDict, is_json_dict, is_object_list

_MAX_STREAM_RESULT_KEYS = 256
_MAX_STREAM_RESULT_DEPTH = 8


def _validate_stream_value(value: object, *, depth: int) -> None:
    """Recursively validate stream result values stay JSON-safe and bounded."""
    if depth > _MAX_STREAM_RESULT_DEPTH:
        raise ValueError("Stream result exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if is_object_list(value):
        if len(value) > _MAX_STREAM_RESULT_KEYS:
            raise ValueError("Stream result list exceeds maximum size")
        for item in value:
            _validate_stream_value(item, depth=depth + 1)
        return
    if is_json_dict(value):
        if len(value) > _MAX_STREAM_RESULT_KEYS:
            raise ValueError("Stream result object exceeds maximum size")
        for nested in value.values():
            _validate_stream_value(nested, depth=depth + 1)
        return
    raise ValueError(f"Unsupported stream result value type: {type(value).__name__}")


def validate_stream_result(result: dict[str, object]) -> JsonDict:
    """Validate and copy a project stream result before resolving a pending request."""
    if not is_json_dict(result):
        raise ValueError("Stream result must be a JSON object")
    if len(result) > _MAX_STREAM_RESULT_KEYS:
        raise ValueError("Stream result exceeds maximum key count")
    for value in result.values():
        _validate_stream_value(value, depth=1)
    return dict(result)


__all__ = ["validate_stream_result"]
