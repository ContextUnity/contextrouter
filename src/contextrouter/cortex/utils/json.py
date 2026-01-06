"""JSON helpers shared across brain nodes."""

from __future__ import annotations


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
