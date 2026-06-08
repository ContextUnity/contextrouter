"""LangGraph state message extraction — last-user-query lookup for node executors."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from contextunity.router.cortex.types import extract_message_content


def get_last_user_query(messages: Sequence[BaseMessage]) -> str:
    """Walk *messages* in reverse and return the text of the last user message."""
    for msg in reversed(messages):
        if msg.type == "human":
            return extract_message_content(msg)
    return ""
