"""Routing functions for dispatcher agent graph edges."""

from __future__ import annotations

from typing import Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_list

from contextunity.router.cortex.types import GraphState

logger = get_contextunit_logger(__name__)


def _has_tool_calls(message: object) -> bool:
    """Return whether a LangChain-style message carries tool calls."""
    tool_calls = getattr(message, "tool_calls", None)
    return is_object_list(tool_calls) and len(tool_calls) > 0


def _message_text(message: object) -> str | None:
    """Extract plain text content when the message exposes a string payload."""
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else None


def should_execute_tools(state: GraphState) -> Literal["execute", "blocked", "end"]:
    """Decide whether to execute tools or handle security violations.

    Called after security_guard_node to determine next step.
    """
    if state["iteration"] >= state.get("max_iterations", 10):
        logger.warning(
            "Max iterations (%d) reached — ending dispatcher loop",
            state.get("max_iterations", 10),
        )
        return "end"

    messages = state["messages"]
    last_message = messages[-1]

    # Security violation occurred
    content = _message_text(last_message)
    if content is not None:
        if "Security Violation" in content or "Error: Tool" in content:
            return "blocked"

    # Has tool calls to execute
    if _has_tool_calls(last_message):
        return "execute"

    return "end"


def should_continue(state: GraphState) -> Literal["tools", "end"]:
    """Decide whether to continue tool execution or end.

    **Unused in the default dispatcher graph** — iteration and tool routing use
    ``should_execute_tools`` after ``security_guard``. Kept for subgraph tests and
    alternate topologies; do not wire after ``tools_executor`` (post-tool messages
    are ``ToolMessage``, not fresh tool calls).
    """
    messages = state["messages"]
    last_message = messages[-1]

    if state["iteration"] >= state.get("max_iterations", 10):
        logger.warning("Max iterations (%d) reached", state.get("max_iterations", 10))
        return "end"

    if _has_tool_calls(last_message):
        return "tools"

    content = _message_text(last_message)
    if content is not None:
        if "final_answer" in content.lower() or not _has_tool_calls(last_message):
            return "end"

    return "end"


__all__ = ["should_continue", "should_execute_tools"]
