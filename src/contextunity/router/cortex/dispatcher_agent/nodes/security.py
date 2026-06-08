"""Security guard node — enforces tool access control."""

from __future__ import annotations

from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.permissions import ToolRisk, ToolScope, check_tool_scope, has_tool_access
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from contextunity.router.cortex.types import (
    BlockedToolCall,
    ConfirmToolCall,
    GraphState,
    SecurityFlag,
    StateUpdate,
)

logger = get_contextunit_logger(__name__)


def _policy_scope_for_tool(tool_name: str) -> str:
    """Pick the policy scope used for HITL/risk classification."""
    from contextunity.core.permissions.policy import DEFAULT_TOOL_POLICIES

    for policy_name in DEFAULT_TOOL_POLICIES:
        if tool_name == policy_name or tool_name.startswith(f"{policy_name}_"):
            return ToolScope.WRITE
    return ToolScope.READ


def _extract_tool_calls(message: object) -> list[dict[str, object]]:
    """Return LangChain tool calls as plain object dictionaries."""
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not is_object_list(raw_tool_calls):
        return []

    tool_calls: list[dict[str, object]] = []
    for item in raw_tool_calls:
        if is_object_dict(item):
            tool_calls.append(dict(item))
    return tool_calls


def _tool_name(tool_call: dict[str, object]) -> str | None:
    """Read a tool name from either ``name`` or legacy ``tool`` keys."""
    name_raw = tool_call.get("name")
    if isinstance(name_raw, str) and name_raw:
        return name_raw
    legacy_name = tool_call.get("tool")
    if isinstance(legacy_name, str) and legacy_name:
        return legacy_name
    return None


def _tool_call_id(tool_call: dict[str, object]) -> str | None:
    """Return the optional LangChain tool call id."""
    raw_id = tool_call.get("id")
    return raw_id if isinstance(raw_id, str) and raw_id else None


def _blocked_tool_messages(blocked_calls: list[BlockedToolCall]) -> list[ToolMessage]:
    """Build LangChain ToolMessages that stop denied tool calls fail-closed."""
    error_messages: list[ToolMessage] = []
    for blocked in blocked_calls:
        tool_name = blocked["tool"]
        reason = blocked["reason"]
        tool_call_id = blocked["tool_call_id"] or "unknown"

        if reason == "tool_not_found":
            error_msg = f"Error: Tool '{tool_name}' is not available in the system."
        elif reason == "tool_denied":
            error_msg = (
                f"Security Violation: Access to tool '{tool_name}' is denied. "
                "This tool is in the denied_tools list."
            )
        elif reason == "confirmation_required":
            error_msg = (
                f"Security Violation: Tool '{tool_name}' requires human confirmation, "
                "but HITL confirmation is unavailable."
            )
        else:
            error_msg = (
                f"Security Violation: Access to tool '{tool_name}' is denied. "
                "Your token does not grant permission for this tool."
            )
        error_messages.append(ToolMessage(tool_call_id=tool_call_id, content=error_msg))
    return error_messages


def _resolve_token_permissions(state: GraphState) -> tuple[str, ...] | None:
    """Read token permissions from secure-node injection or legacy access_token."""
    state_map: dict[str, object] = dict(state)
    token_obj = state_map.get("__token__")
    if token_obj is None:
        token_obj = state_map.get("access_token")
    if not isinstance(token_obj, ContextToken):
        return None
    return token_obj.permissions


async def security_guard_node(state: GraphState, config: RunnableConfig) -> StateUpdate:
    """Check requested tool calls against discovery, permissions, and risk policy."""
    _ = config
    last_message = state["messages"][-1]
    tool_calls = _extract_tool_calls(last_message)
    if not tool_calls:
        return {}

    # Discover and filter tools
    from contextunity.router.cortex.dispatcher_agent.tool_resolution import (
        dispatcher_available_tools_for_state,
    )

    available_tool_names = {tool.name for tool in dispatcher_available_tools_for_state(state)}
    token_permissions = _resolve_token_permissions(state)
    allowed_tools = state.get("allowed_tools", ["*"])
    denied_tools = state.get("denied_tools", [])

    blocked_calls: list[BlockedToolCall] = []
    security_flags: list[SecurityFlag] = list(state.get("security_flags", []))
    confirm_required: list[ConfirmToolCall] = []

    for tool_call in tool_calls:
        tool_name = _tool_name(tool_call)
        tool_call_id = _tool_call_id(tool_call)
        if tool_name is None:
            blocked_calls.append(
                {
                    "tool": "unknown",
                    "reason": "tool_not_found",
                    "tool_call_id": tool_call_id,
                }
            )
            security_flags.append({"event": "tool_not_found", "tool": "unknown"})
            logger.warning("Security: Tool call missing name/tool field")
            continue

        if tool_name not in available_tool_names:
            blocked_calls.append(
                {"tool": tool_name, "reason": "tool_not_found", "tool_call_id": tool_call_id}
            )
            security_flags.append({"event": "tool_not_found", "tool": tool_name})
            logger.warning("Security: Tool '%s' not found", tool_name)
            continue

        if denied_tools and tool_name in denied_tools:
            blocked_calls.append(
                {"tool": tool_name, "reason": "tool_denied", "tool_call_id": tool_call_id}
            )
            security_flags.append({"event": "tool_denied", "tool": tool_name})
            logger.warning("Security: Tool '%s' in denied_tools", tool_name)
            continue

        if token_permissions is None:
            blocked_calls.append(
                {"tool": tool_name, "reason": "permission_denied", "tool_call_id": tool_call_id}
            )
            security_flags.append({"event": "permission_denied", "tool": tool_name})
            logger.warning(
                "Security: Tool '%s' blocked — token required for tool execution",
                tool_name,
            )
            continue
        if not has_tool_access(token_permissions, tool_name):
            blocked_calls.append(
                {
                    "tool": tool_name,
                    "reason": "permission_denied",
                    "tool_call_id": tool_call_id,
                }
            )
            security_flags.append({"event": "permission_denied", "tool": tool_name})
            logger.warning("Security: Token denies access to tool '%s'", tool_name)
            continue
        if "*" not in allowed_tools and tool_name not in allowed_tools:
            blocked_calls.append(
                {"tool": tool_name, "reason": "permission_denied", "tool_call_id": tool_call_id}
            )
            security_flags.append({"event": "permission_denied", "tool": tool_name})
            logger.warning("Security: Tool '%s' not in allowed_tools", tool_name)
            continue

        policy_scope = _policy_scope_for_tool(tool_name)
        run_policy_check = True
        policy_perms = token_permissions

        if run_policy_check:
            try:
                risk = check_tool_scope(policy_perms, tool_name, policy_scope)
            except (TypeError, ValueError) as exc:
                blocked_calls.append(
                    {
                        "tool": tool_name,
                        "reason": "permission_denied",
                        "tool_call_id": tool_call_id,
                    }
                )
                security_flags.append({"event": "permission_denied", "tool": tool_name})
                logger.warning("Security: Policy check failed for tool '%s': %s", tool_name, exc)
                continue

            if risk == ToolRisk.DENY:
                blocked_calls.append(
                    {
                        "tool": tool_name,
                        "reason": "permission_denied",
                        "tool_call_id": tool_call_id,
                    }
                )
                security_flags.append({"event": "permission_denied", "tool": tool_name})
                logger.warning(
                    "Security: Policy denies tool '%s' at %s scope",
                    tool_name,
                    policy_scope,
                )
                continue

            if risk == ToolRisk.CONFIRM:
                confirm_required.append(
                    {
                        "tool": tool_name,
                        "tool_call_id": tool_call_id,
                        "risk": "CONFIRM",
                        "reason": "requires human confirmation",
                    }
                )
                security_flags.append({"event": "hitl_confirm_required", "tool": tool_name})
                logger.info("Security: Tool '%s' requires HITL confirmation", tool_name)
                continue

        logger.debug("Security: Tool '%s' allowed", tool_name)

    if confirm_required and not state.get("hitl_approved"):
        try:
            from langgraph.types import interrupt

            tool_names = [entry["tool"] for entry in confirm_required]
            interrupt(
                {
                    "type": "tool_confirmation",
                    "tools": confirm_required,
                    "message": (
                        f"The following tools require human approval before execution: {tool_names}"
                    ),
                }
            )
        except ImportError:
            logger.warning("langgraph.types.interrupt not available — blocking HITL tools")
            for confirm in confirm_required:
                blocked_calls.append(
                    {
                        "tool": confirm["tool"],
                        "reason": "confirmation_required",
                        "tool_call_id": confirm["tool_call_id"],
                    }
                )

    if blocked_calls:
        blocked_names = [blocked["tool"] for blocked in blocked_calls]
        logger.warning(
            "Security violation: %d tool(s) blocked. Blocked: %s",
            len(blocked_calls),
            blocked_names,
        )
        return {"messages": _blocked_tool_messages(blocked_calls), "security_flags": security_flags}

    return {"security_flags": security_flags} if security_flags else {}


__all__ = ["security_guard_node"]
