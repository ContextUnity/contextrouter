"""JSON helpers -- safe parsing, extraction, and repair for LLM-generated JSON outputs."""

from __future__ import annotations

from typing import TypeVar, overload

from contextunity.core.parsing import json_loads
from contextunity.core.types import WireValue

_DefaultT = TypeVar("_DefaultT")


def strip_json_fence(text: str) -> str:
    """Remove common markdown code fences (e.g. ```json ... ```) from LLM output.

    Args:
        text: The raw text string containing potential JSON content inside code fences.

    Returns:
        The stripped string content with code fences and leading/trailing whitespace removed.
    """
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    # Strip leading/trailing backticks and optional language header.
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].lstrip()
    return raw.strip()


@overload
def safe_json_loads(text: str) -> WireValue | None: ...


@overload
def safe_json_loads(text: str, default: _DefaultT) -> WireValue | _DefaultT: ...


def safe_json_loads(text: str, default: _DefaultT | None = None) -> WireValue | _DefaultT | None:
    """Safely parse a JSON string, stripping markdown fences and returning a default value on failure.

    Args:
        text: The string containing JSON formatted data.
        default: The fallback value to return if parsing fails. Defaults to None.

    Returns:
        The parsed Python object (dict, list, etc.) or the default value on error.
    """
    if not text:
        return default
    try:
        cleaned = strip_json_fence(text)
        return json_loads(cleaned)
    except ValueError:
        return default
