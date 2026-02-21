"""Routing functions for dispatcher agent graph edges."""

from __future__ import annotations

import logging
from typing import Literal

from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState

logger = logging.getLogger(__name__)


def should_execute_tools(state: DispatcherState) -> Literal["execute", "blocked", "end"]:
    """Decide whether to execute tools or handle security violations.

    Called after security_guard_node to determine next step.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Security violation occurred
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        if "Security Violation" in last_message.content or "Error: Tool" in last_message.content:
            return "blocked"

    # Has tool calls to execute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute"

    return "end"


def should_continue(state: DispatcherState) -> Literal["tools", "end"]:
    """Decide whether to continue tool execution or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if state["iteration"] >= state.get("max_iterations", 10):
        logger.warning("Max iterations (%d) reached", state.get("max_iterations", 10))
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        if "final_answer" in last_message.content.lower() or not last_message.tool_calls:
            return "end"

    return "end"


__all__ = ["should_continue", "should_execute_tools"]
