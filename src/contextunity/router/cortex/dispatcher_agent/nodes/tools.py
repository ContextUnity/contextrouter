"""Dispatcher tool execution node — runs approved tool calls inside secure_node."""

from __future__ import annotations

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from contextunity.router.cortex.types import GraphState, StateUpdate
from contextunity.router.langchain_boundaries import invoke_runnable_ainvoke

logger = get_contextunit_logger(__name__)


async def tools_executor_node(state: GraphState, config: RunnableConfig) -> StateUpdate:
    """Execute tool calls from the last agent message using LangGraph ToolNode."""
    from contextunity.router.cortex.dispatcher_agent.tool_resolution import (
        dispatcher_tools_for_state,
    )

    _ = config
    tools = dispatcher_tools_for_state(state)

    if not tools:
        logger.warning("Dispatcher tools node invoked with no executable tools")
        return {}

    tool_node = ToolNode(tools)
    tool_payload_obj = await invoke_runnable_ainvoke(tool_node, state, config=config)
    if is_object_dict(tool_payload_obj):
        return tool_payload_obj
    return {}


__all__ = ["tools_executor_node"]
