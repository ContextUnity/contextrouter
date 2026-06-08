"""Shared dispatcher tool allow-list resolution for agent and tools nodes."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from contextunity.router.cortex.config_resolution import metadata_project_id
from contextunity.router.cortex.types import GraphState


def dispatcher_available_tools_for_state(state: GraphState) -> list[BaseTool]:
    """Resolve tools visible to the trusted project on dispatcher state."""
    from contextunity.router.modules.tools import discover_tools_for_project

    return discover_tools_for_project(metadata_project_id(state))


def dispatcher_tools_for_state(state: GraphState) -> list[BaseTool]:
    """Resolve executable tools from ``allowed_tools`` on dispatcher state.

    Contract (see ``ExecuteDispatcherPayload``):
      - empty list → no tools
      - ``[\"*\"]`` → all discovered tools
      - non-empty names → filtered subset
    """
    tools = dispatcher_available_tools_for_state(state)
    allowed = state.get("allowed_tools", [])
    if "*" in allowed:
        return tools
    if not allowed:
        return []
    allowed_set = set(allowed)
    return [tool for tool in tools if tool.name in allowed_set]


__all__ = ["dispatcher_available_tools_for_state", "dispatcher_tools_for_state"]
