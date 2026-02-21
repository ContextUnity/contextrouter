"""Security guard node — enforces tool access control."""

from __future__ import annotations

import logging
from typing import Any

from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState
from contextrouter.modules.tools import discover_all_tools

logger = logging.getLogger(__name__)


async def security_guard_node(state: DispatcherState) -> dict[str, Any]:
    """Security guard node — checks tool permissions before execution.

    Enforcement layers (in priority order):
    1. Tool must exist in discover_all_tools()
    2. Token-based: ``has_tool_access(token.permissions, tool_name)``
    3. State-based fallback: ``denied_tools`` / ``allowed_tools`` lists
    """
    from langchain_core.messages import ToolMessage

    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    available_tools = discover_all_tools()
    available_tool_names = {tool.name for tool in available_tools}

    # Extract permissions from token
    token = state.get("access_token")
    token_permissions: tuple[str, ...] | None = None
    if token is not None and hasattr(token, "permissions"):
        token_permissions = token.permissions

    allowed_tools = state.get("allowed_tools", [])
    denied_tools = state.get("denied_tools", [])

    blocked_calls = []
    security_flags: list[dict[str, Any]] = list(state.get("security_flags", []))
    confirm_required: list[dict[str, Any]] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name") or tool_call.get("tool")
        if not tool_name:
            continue

        # 1. Check if tool exists
        if tool_name not in available_tool_names:
            blocked_calls.append(
                {
                    "tool": tool_name,
                    "reason": "tool_not_found",
                    "tool_call_id": tool_call.get("id"),
                }
            )
            security_flags.append({"event": "tool_not_found", "tool": tool_name})
            logger.warning("Security: Tool '%s' not found", tool_name)
            continue

        # 2. Check blacklist
        if denied_tools and tool_name in denied_tools:
            blocked_calls.append(
                {
                    "tool": tool_name,
                    "reason": "tool_denied",
                    "tool_call_id": tool_call.get("id"),
                }
            )
            security_flags.append({"event": "tool_denied", "tool": tool_name})
            logger.warning("Security: Tool '%s' in denied_tools", tool_name)
            continue

        # 3. Token-based permission check
        if token_permissions is not None:
            try:
                from contextcore.permissions import has_tool_access

                if not has_tool_access(token_permissions, tool_name):
                    blocked_calls.append(
                        {
                            "tool": tool_name,
                            "reason": "permission_denied",
                            "tool_call_id": tool_call.get("id"),
                        }
                    )
                    security_flags.append({"event": "permission_denied", "tool": tool_name})
                    logger.warning("Security: Token denies access to tool '%s'", tool_name)
                    continue
            except ImportError:
                logger.debug("contextcore.permissions not available, falling back")

        # 4. State-level whitelist fallback
        elif allowed_tools:
            if "*" not in allowed_tools and tool_name not in allowed_tools:
                blocked_calls.append(
                    {
                        "tool": tool_name,
                        "reason": "permission_denied",
                        "tool_call_id": tool_call.get("id"),
                    }
                )
                security_flags.append({"event": "permission_denied", "tool": tool_name})
                logger.warning("Security: Tool '%s' not in allowed_tools", tool_name)
                continue

        # 5. Tool scope / risk check (HITL for CONFIRM-risk tools)
        try:
            from contextcore.permissions import ToolRisk, check_tool_scope

            perms = token_permissions or ("*",)
            risk = check_tool_scope(perms, tool_name, "execute")
            if risk == ToolRisk.CONFIRM:
                confirm_required.append(
                    {
                        "tool": tool_name,
                        "tool_call_id": tool_call.get("id"),
                        "risk": "CONFIRM",
                        "reason": "requires human confirmation",
                    }
                )
                security_flags.append({"event": "hitl_confirm_required", "tool": tool_name})
                logger.info("Security: Tool '%s' requires HITL confirmation", tool_name)
        except (ImportError, TypeError):
            pass

        logger.debug("Security: Tool '%s' allowed", tool_name)

    # HITL interrupt
    if confirm_required and not state.get("hitl_approved"):
        try:
            from langgraph.types import interrupt

            interrupt(
                {
                    "type": "tool_confirmation",
                    "tools": confirm_required,
                    "message": (
                        f"The following tools require human approval before execution: "
                        f"{[c['tool'] for c in confirm_required]}"
                    ),
                }
            )
        except ImportError:
            logger.warning("langgraph.types.interrupt not available — skipping HITL")

    # Return error messages for blocked tools
    if blocked_calls:
        error_messages = []
        for blocked in blocked_calls:
            tool_name = blocked["tool"]
            reason = blocked["reason"]
            tool_call_id = blocked.get("tool_call_id", "unknown")

            if reason == "tool_not_found":
                error_msg = f"Error: Tool '{tool_name}' is not available in the system."
            elif reason == "tool_denied":
                error_msg = (
                    f"Security Violation: Access to tool '{tool_name}' is denied. "
                    f"This tool is in the denied_tools list."
                )
            elif reason == "permission_denied":
                error_msg = (
                    f"Security Violation: Access to tool '{tool_name}' is denied. "
                    f"Your token does not grant permission for this tool."
                )
            else:
                error_msg = f"Security Violation: Access to tool '{tool_name}' is denied."

            error_messages.append(ToolMessage(tool_call_id=tool_call_id, content=error_msg))

        logger.warning(
            "Security violation: %d tool(s) blocked. Blocked: %s",
            len(blocked_calls),
            [b["tool"] for b in blocked_calls],
        )
        return {"messages": error_messages, "security_flags": security_flags}

    return {"security_flags": security_flags} if security_flags else {}


__all__ = ["security_guard_node"]
