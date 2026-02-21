"""Runner entrypoint for the dispatcher agent.

This module provides the main entry points for invoking the always-active
dispatcher agent, similar to the chat runner but for the dispatcher.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Sequence

from contextcore import ContextToken
from langchain_core.messages import BaseMessage, HumanMessage

from contextrouter.cortex.services.dispatcher import get_dispatcher_service

logger = logging.getLogger(__name__)


async def invoke_dispatcher(
    messages: Sequence[BaseMessage] | list[dict[str, Any]],
    tenant_id: str = "default",
    session_id: str = "default",
    platform: str = "api",
    metadata: dict[str, Any] | None = None,
    max_iterations: int = 10,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    trace_id: str | None = None,
    access_token: ContextToken | None = None,
) -> dict[str, Any]:
    """Invoke the dispatcher agent (non-streaming).

    Args:
        messages: List of messages (LangChain BaseMessage or dict format)
        tenant_id: Tenant identifier for multi-tenant isolation
        session_id: Session identifier
        platform: Platform identifier
        metadata: Additional metadata
        max_iterations: Maximum number of agent iterations

    Returns:
        Final state from graph execution
    """
    # Convert dict messages to BaseMessage if needed
    if messages and isinstance(messages[0], dict):
        lc_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                else:
                    from langchain_core.messages import AIMessage

                    lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(msg)
        messages = lc_messages

    service = get_dispatcher_service()
    return await service.invoke(
        messages=list(messages),
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
    messages: Sequence[BaseMessage] | list[dict[str, Any]],
    tenant_id: str = "default",
    session_id: str = "default",
    platform: str = "api",
    metadata: dict[str, Any] | None = None,
    max_iterations: int = 10,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    trace_id: str | None = None,
    access_token: ContextToken | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream results from the dispatcher agent.

    Args:
        messages: List of messages (LangChain BaseMessage or dict format)
        tenant_id: Tenant identifier for multi-tenant isolation
        session_id: Session identifier
        platform: Platform identifier
        metadata: Additional metadata
        max_iterations: Maximum number of agent iterations

    Yields:
        Events from graph execution
    """
    # Convert dict messages to BaseMessage if needed
    if messages and isinstance(messages[0], dict):
        lc_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                else:
                    from langchain_core.messages import AIMessage

                    lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(msg)
        messages = lc_messages

    service = get_dispatcher_service()
    async for event in service.stream(
        messages=list(messages),
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
