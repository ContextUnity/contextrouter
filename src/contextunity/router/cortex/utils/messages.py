"""Message helpers shared across cortex graph nodes."""

from __future__ import annotations

from collections.abc import Sequence

from contextunity.core.types import is_object_dict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from contextunity.router.cortex.types import MessageDict


def get_last_human_text(messages: Sequence[BaseMessage] | None) -> str:
    """Return the text content of the most recent human message in the history.

    Args:
        messages: A sequence of message objects or dictionaries representing the chat history.

    Returns:
        The stripped string content of the last user query, or an empty string if not found.
    """
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.text.strip()
        if is_object_dict(msg):
            role = msg.get("role")
            msg_type = msg.get("type")
            if role == "user" or role == "human" or msg_type == "human":
                return _stringify_content(msg.get("content")).strip()
    return ""


def format_conversation_history(
    messages: Sequence[BaseMessage | MessageDict] | None,
    *,
    last_n: int = 10,
    max_content_len: int = 500,
    exclude_last: bool = False,
) -> str:
    """Format recent conversation messages as simplified ``Role: content`` text lines.

    Extracts text content and normalizes whitespace for each message in the specified window,
    useful for formatting context to inject into LLM prompts.

    Args:
        messages: Sequence of LangChain messages or serialized dicts.
        last_n: Number of most recent messages to include. Defaults to 10.
        max_content_len: Maximum characters allowed per message content. Defaults to 500.
        exclude_last: If True, exclude the final message in the sequence (useful when
            the caller already processes the current user query separately). Defaults to False.

    Returns:
        Newline-joined string of formatted ``User: ...`` / ``Assistant: ...`` lines.
    """
    if not messages:
        return ""
    window = list(messages)
    if exclude_last and window:
        window = window[:-1]
    window = window[-last_n:]
    parts: list[str] = []
    for msg in window:
        role = _resolve_role(msg)
        content = _resolve_content(msg)
        content = " ".join((content or "").split())  # normalise whitespace
        if content:
            parts.append(f"{role}: {content[:max_content_len]}")
    return "\n".join(parts)


def _resolve_role(msg: BaseMessage | MessageDict) -> str:
    """Determine whether the message role is 'User' or 'Assistant'.

    Args:
        msg: A message object or a dictionary representation.

    Returns:
        The resolved role name ('User' or 'Assistant').
    """
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Assistant"
    if is_object_dict(msg):
        role = msg.get("role")
        if role == "user" or role == "human" or msg.get("type") == "human":
            return "User"
        return "Assistant"
    msg_type = getattr(msg, "type", "")
    return "User" if msg_type == "human" else "Assistant"


def _resolve_content(msg: BaseMessage | MessageDict) -> str:
    """Extract string content from a message object or dictionary.

    Args:
        msg: A message object or a dictionary representation.

    Returns:
        The string content of the message.
    """
    if is_object_dict(msg):
        raw = msg.get("content", "")
    else:
        raw = getattr(msg, "content", str(msg))
    return _stringify_content(raw)


def _stringify_content(raw: object) -> str:
    """Convert message content to display text."""
    return raw if isinstance(raw, str) else str(raw or "")
