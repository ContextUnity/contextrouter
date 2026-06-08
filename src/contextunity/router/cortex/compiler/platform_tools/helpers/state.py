"""Shared state normalization helpers for platform tools."""

from __future__ import annotations

from contextunity.router.cortex.compiler.state_routing import read_state_input
from contextunity.router.cortex.types import extract_message_content

from .contracts import PlatformState


def as_text(value: object) -> str:
    """Normalize heterogeneous state values into text."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def get_text_from_state(
    state: PlatformState,
    primary_key: str,
    *,
    fallback_key: str | None = None,
) -> str:
    """Read primary/fallback state key and normalize to text."""
    primary: object = read_state_input(state, primary_key)
    if primary:
        return as_text(primary)
    if fallback_key:
        fallback_value: object = read_state_input(state, fallback_key)
        return as_text(fallback_value)
    return ""


def get_last_message_text(state: PlatformState) -> str:
    """Get last message content from state['messages'] if present."""
    messages = state.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return ""
    return extract_message_content(messages[-1])


__all__ = ["as_text", "get_last_message_text", "get_text_from_state"]
