"""Builder for the dispatcher agent graph.

Assembles nodes into the LangGraph pipeline:

    agent → security → [execute|blocked|end]
                         → execute → tools → agent
                         → blocked → agent (with error)
                         → end → reflect → END
"""

from __future__ import annotations

from contextunity.core import get_contextunit_logger
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END

from contextunity.router.cortex.dispatcher_agent.nodes import (
    agent_node,
    reflect_dispatcher,
    security_guard_node,
    tools_executor_node,
)
from contextunity.router.cortex.dispatcher_agent.routing import should_execute_tools
from contextunity.router.cortex.dispatcher_agent.types import DispatcherState
from contextunity.router.cortex.types import CortexGraph, RunnableGraph

logger = get_contextunit_logger(__name__)


def build_dispatcher_graph() -> CortexGraph[DispatcherState]:
    """Build the dispatcher agent graph with security guard and all tools.

    Graph flow:
        agent → security → [execute|blocked|end]
                             → execute → tools → agent
                             → blocked → agent (with error)
                             → end → reflect → END
    """

    from contextunity.router.cortex.secure_node import make_secure_node
    from contextunity.router.modules.tools import discover_all_tools

    tools = discover_all_tools()
    logger.debug("Building dispatcher graph with %d tools", len(tools))

    workflow = CortexGraph(DispatcherState)

    from contextunity.router.cortex.compiler.types import CompilerNodeSpec

    dispatcher_spec: CompilerNodeSpec = {}
    tool_names = [tool.name for tool in tools]

    secure_agent = make_secure_node("agent", agent_node, dispatcher_spec)
    secure_reflect = make_secure_node("reflect", reflect_dispatcher)
    secure_tools = (
        make_secure_node(
            "tools",
            tools_executor_node,
            dispatcher_spec,
            requires_llm=False,
            execute_tools=tool_names,
        )
        if tools
        else None
    )

    workflow.add_typed_node("agent", secure_agent)
    workflow.add_typed_node("security", security_guard_node)
    workflow.add_typed_node("reflect", secure_reflect)

    if secure_tools:
        workflow.add_typed_node("tools", secure_tools)

    workflow.set_typed_entry_point("agent")
    workflow.add_typed_edge("agent", "security")

    if secure_tools:
        workflow.add_typed_conditional_edges(
            "security",
            should_execute_tools,
            {
                "execute": "tools",
                "blocked": "agent",
                "end": "reflect",
            },
        )
        workflow.add_typed_edge("tools", "agent")
    else:
        workflow.add_typed_conditional_edges(
            "security",
            should_execute_tools,
            {
                "execute": "reflect",
                "blocked": "agent",
                "end": "reflect",
            },
        )

    workflow.add_typed_edge("reflect", END)

    return workflow


def compile_dispatcher_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> RunnableGraph:
    """Build and compile the dispatcher graph, optionally attaching a *checkpointer* for state persistence."""
    workflow = build_dispatcher_graph()
    if checkpointer is None:
        return workflow.compile_typed()
    return workflow.compile_typed(checkpointer=checkpointer)


__all__ = ["build_dispatcher_graph", "compile_dispatcher_graph"]
