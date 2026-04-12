"""JSON helpers shared across brain nodes."""

from __future__ import annotations

import json
from typing import Any


def strip_json_fence(text: str) -> str:
    """Remove common ```json fences from LLM output (best-effort)."""
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    # Strip leading/trailing backticks and optional language header.
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].lstrip()
    return raw.strip()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure.

    Also handles ```json fences from LLM output.
    """
    if not text:
        return default
    try:
        cleaned = strip_json_fence(text)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return default
