"""Message helpers shared across brain nodes."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage


def get_last_human_text(messages: Sequence[BaseMessage] | None) -> str:
    """Return the most recent human message text (best-effort)."""
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            return (content if isinstance(content, str) else str(content)).strip()

        # Some LangGraph / serialized message variants appear as dicts.
        if isinstance(msg, dict) and msg.get("type") == "human":
            content = msg.get("content")
            return (content if isinstance(content, str) else str(content or "")).strip()

    return ""
