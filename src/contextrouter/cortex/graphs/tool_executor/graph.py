"""Tool Executor Proxy Graph.

This acts as a 'graph' for ExecuteAgent when agent_id='tool_executor'.
It proxies the tool execution request to the connected BiDi stream
via StreamExecutorManager.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from contextrouter.core.registry import register_graph
from contextrouter.service.stream_executors import get_stream_executor_manager


async def execute_tool_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the tool via BiDi stream."""
    manager = get_stream_executor_manager()
    tool_name = state.get("tool")
    args = state.get("args", {})
    target_project = state.get("target_project")

    if not tool_name or not target_project:
        raise ValueError("Both 'tool' and 'target_project' are required for tool_executor.")

    # Call the project stream and wait for result
    result = await manager.execute(target_project, tool_name, args)

    # Return the full result dictated by the connected project
    # StateGraph update will replace/merge this into the state dictionary
    return result


@register_graph("tool_executor")
def build_tool_executor_graph() -> Any:
    """Build the ToolExecutorProxy pseudo-graph."""
    builder = StateGraph(dict)
    builder.add_node("tool_executor", execute_tool_node)
    builder.add_edge(START, "tool_executor")
    builder.add_edge("tool_executor", END)
    return builder.compile()
