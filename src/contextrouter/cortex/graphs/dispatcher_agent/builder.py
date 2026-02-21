"""Builder for the dispatcher agent graph.

Assembles nodes into the LangGraph pipeline:

    agent → security → [execute|blocked|end]
                         → execute → tools → agent
                         → blocked → agent (with error)
                         → end → reflect → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from contextrouter.cortex.graphs.dispatcher_agent.nodes import (
    agent_node,
    reflect_dispatcher,
    security_guard_node,
)
from contextrouter.cortex.graphs.dispatcher_agent.routing import should_execute_tools
from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState
from contextrouter.modules.tools import discover_all_tools

logger = logging.getLogger(__name__)


def build_dispatcher_graph() -> StateGraph:
    """Build the dispatcher agent graph with security guard and all tools.

    Graph flow:
        agent → security → [execute|blocked|end]
                             → execute → tools → agent
                             → blocked → agent (with error)
                             → end → reflect → END
    """
    tools = discover_all_tools()
    logger.info("Building dispatcher graph with %d tools", len(tools))

    tool_node = ToolNode(tools) if tools else None

    workflow = StateGraph(DispatcherState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("security", security_guard_node)
    workflow.add_node("reflect", reflect_dispatcher)

    if tool_node:
        workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "security")

    if tool_node:
        workflow.add_conditional_edges(
            "security",
            should_execute_tools,
            {
                "execute": "tools",
                "blocked": "agent",
                "end": "reflect",
            },
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_conditional_edges(
            "security",
            should_execute_tools,
            {
                "execute": "reflect",
                "blocked": "agent",
                "end": "reflect",
            },
        )

    workflow.add_edge("reflect", END)

    return workflow


def compile_dispatcher_graph(checkpointer: Any | None = None) -> Any:
    """Compile and return the dispatcher graph.

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
                     If None, no checkpointing is used.

    Returns:
        Compiled LangGraph instance.
    """
    workflow = build_dispatcher_graph()
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


__all__ = ["build_dispatcher_graph", "compile_dispatcher_graph"]
