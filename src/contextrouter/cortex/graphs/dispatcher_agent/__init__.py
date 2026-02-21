"""Dispatcher Agent Graph — always-active agent with all tools connected.

This graph provides a persistent agent that:
- Automatically discovers and connects all available tools
- Routes requests to appropriate tools based on context
- Enforces security via token + allowlist/denylist checks
- Logs execution traces to Brain

Usage:
    from contextrouter.cortex.graphs.dispatcher_agent import compile_dispatcher_graph

    graph = compile_dispatcher_graph(checkpointer=saver)
    result = await graph.ainvoke(initial_state)
"""

from contextrouter.cortex.graphs.dispatcher_agent.builder import (
    build_dispatcher_graph,
    compile_dispatcher_graph,
)
from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState

__all__ = ["DispatcherState", "build_dispatcher_graph", "compile_dispatcher_graph"]
