"""Runner entrypoint for the dispatcher agent.

This module provides the main entry points for invoking the always-active
dispatcher agent, similar to the chat runner but for the dispatcher.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.exceptions import SecurityError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from contextunity.router.cortex.services.dispatcher import get_dispatcher_service

from .dispatcher_agent.types import DispatcherState
from .types import ExecutionMetadata, MessageDict, StateUpdate

logger = get_contextunit_logger(__name__)


def _to_base_messages(
    messages: Sequence[BaseMessage] | list[MessageDict],
) -> list[BaseMessage]:
    """Normalize incoming messages to a flat list of LangChain BaseMessage objects."""
    if not messages:
        return []

    # Already BaseMessage — pass through
    if isinstance(messages[0], BaseMessage):
        return [m for m in messages if isinstance(m, BaseMessage)]

    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                result.append(HumanMessage(content=content))
            else:
                result.append(AIMessage(content=content))
    return result


async def invoke_dispatcher(
    messages: Sequence[BaseMessage] | list[MessageDict],
    tenant_id: str = "default",
    session_id: str = "default",
    platform: str = "api",
    metadata: ExecutionMetadata | None = None,
    max_iterations: int = 10,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    trace_id: str | None = None,
    access_token: ContextToken | None = None,
) -> DispatcherState:
    """Invoke the dispatcher agent synchronously to obtain the final execution state.

    Args:
        messages: Sequence of conversation messages in LangChain BaseMessage format or
            standard MessageDict structures containing 'role' and 'content' keys.
        tenant_id: Tenant identifier for multi-tenant data and policy isolation.
        session_id: The session or conversation identifier for state persistence.
        platform: The client platform identifier (e.g. 'api', 'slack', 'telegram').
        metadata: Execution context metadata to seed into the initial dispatcher state.
        max_iterations: Maximum loop iterations the dispatcher can run before terminating.
        allowed_tools: List of tool names that the dispatcher is explicitly permitted to invoke.
        denied_tools: List of tool names that the dispatcher is prohibited from invoking.
        trace_id: Optional UUID or identifier to correlate sub-graph calls for observability.
        access_token: Authoritative ContextToken containing security context and scopes.

    Returns:
        The final state dict of the dispatcher agent after completion of all execution steps.

    Raises:
        SecurityError: If `access_token` is missing or invalid.
    """
    if access_token is None:
        raise SecurityError("access_token is required")

    from contextunity.router.service.mixins.execution.helpers import (
        resolve_dispatcher_tenant_id,
    )

    tenant_id = resolve_dispatcher_tenant_id(tenant_id, access_token)

    lc_messages = _to_base_messages(messages)
    service = get_dispatcher_service()
    return await service.invoke(
        messages=lc_messages,
        tenant_id=tenant_id,
        session_id=session_id,
        platform=platform,
        metadata=metadata,
        max_iterations=max_iterations,
        allowed_tools=allowed_tools or [],
        denied_tools=denied_tools or [],
        trace_id=trace_id,
        access_token=access_token,
    )


async def stream_dispatcher(
    messages: Sequence[BaseMessage] | list[MessageDict],
    tenant_id: str = "default",
    session_id: str = "default",
    platform: str = "api",
    metadata: ExecutionMetadata | None = None,
    max_iterations: int = 10,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    trace_id: str | None = None,
    access_token: ContextToken | None = None,
) -> AsyncIterator[StateUpdate]:
    """Stream execution updates and events from the dispatcher agent.

    Args:
        messages: Sequence of conversation messages in LangChain BaseMessage format or
            standard MessageDict structures containing 'role' and 'content' keys.
        tenant_id: Tenant identifier for multi-tenant data and policy isolation.
        session_id: The session or conversation identifier for state persistence.
        platform: The client platform identifier (e.g. 'api', 'slack', 'telegram').
        metadata: Execution context metadata to seed into the initial dispatcher state.
        max_iterations: Maximum loop iterations the dispatcher can run before terminating.
        allowed_tools: List of tool names that the dispatcher is explicitly permitted to invoke.
        denied_tools: List of tool names that the dispatcher is prohibited from invoking.
        trace_id: Optional UUID or identifier to correlate sub-graph calls for observability.
        access_token: Authoritative ContextToken containing security context and scopes.

    Yields:
        StateUpdate dictionaries containing step outcomes, tool invocations, or intermediate messages.

    Raises:
        SecurityError: If `access_token` is missing or invalid.
    """
    if access_token is None:
        raise SecurityError("access_token is required")

    from contextunity.router.service.mixins.execution.helpers import (
        resolve_dispatcher_tenant_id,
    )

    tenant_id = resolve_dispatcher_tenant_id(tenant_id, access_token)

    lc_messages = _to_base_messages(messages)
    service = get_dispatcher_service()
    async for event in service.stream(
        messages=lc_messages,
        tenant_id=tenant_id,
        session_id=session_id,
        platform=platform,
        metadata=metadata,
        max_iterations=max_iterations,
        allowed_tools=allowed_tools or [],
        denied_tools=denied_tools or [],
        trace_id=trace_id,
        access_token=access_token,
    ):
        yield event


__all__ = [
    "invoke_dispatcher",
    "stream_dispatcher",
]
