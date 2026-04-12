"""RAG cortex - LangGraph agent implementation.

This package provides the LangGraph-based RAG agent for knowledge-driven question answering.

Usage:
    from contextunity.router.cortex import compile_graph, invoke_agent, stream_agent
    from contextunity.router.modules.observability import get_langfuse_callbacks

    # Compile and invoke with tracing
    graph = compile_graph()
    callbacks = get_langfuse_callbacks(session_id="my_session", user_id="user123")
    result = graph.invoke(input_state, config={"callbacks": callbacks})

    # Or use the runner helpers (tracing included automatically)
    result = await invoke_agent(messages, session_id, platform)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextunity.router.cortex.graphs.brain import build_graph, compile_graph, reset_graph
    from contextunity.router.cortex.runners.ingestion import invoke_ingestion, stream_ingestion

    from contextunity.router.cortex.models import Citation, RetrievedDoc
    from contextunity.router.cortex.runners.chat import (
        create_input_from_query,
        invoke_agent,
        invoke_agent_sync,
        stream_agent,
    )
    from contextunity.router.cortex.services import get_graph_service
    from contextunity.router.cortex.state import (
        AgentState,
        InputState,
        OutputState,
        get_last_user_query,
    )

__all__ = [
    "AgentState",
    "InputState",
    "OutputState",
    "get_last_user_query",
    "Citation",
    "RetrievedDoc",
    "get_graph_service",
    "build_graph",
    "compile_graph",
    "create_input_from_query",
    "invoke_agent",
    "invoke_agent_sync",
    "reset_graph",
    "stream_agent",
    # Ingestion runner
    "invoke_ingestion",
    "stream_ingestion",
    # Dispatcher runner
    "invoke_dispatcher",
    "stream_dispatcher",
]


_EXPORTS: dict[str, str] = {
    # State
    "AgentState": "contextunity.router.cortex.state.AgentState",
    "InputState": "contextunity.router.cortex.state.InputState",
    "OutputState": "contextunity.router.cortex.state.OutputState",
    "get_last_user_query": "contextunity.router.cortex.state.get_last_user_query",
    # Models
    "Citation": "contextunity.router.cortex.models.Citation",
    "RetrievedDoc": "contextunity.router.cortex.models.RetrievedDoc",
    # Services
    "get_graph_service": "contextunity.router.cortex.services.get_graph_service",
    # Graph
    "build_graph": "contextunity.router.cortex.graphs.brain.build_graph",
    "compile_graph": "contextunity.router.cortex.graphs.brain.compile_graph",
    "compile_graph_from_recipe": "contextunity.router.cortex.graphs.rag_ingestion.compile_graph_from_recipe",
    "reset_graph": "contextunity.router.cortex.graphs.brain.reset_graph",
    # Runner
    "create_input_from_query": "contextunity.router.cortex.runners.chat.create_input_from_query",
    "invoke_agent": "contextunity.router.cortex.runners.chat.invoke_agent",
    "invoke_agent_sync": "contextunity.router.cortex.runners.chat.invoke_agent_sync",
    "stream_agent": "contextunity.router.cortex.runners.chat.stream_agent",
    # Ingestion runner
    "invoke_ingestion": "contextunity.router.cortex.runners.ingestion.invoke_ingestion",
    "stream_ingestion": "contextunity.router.cortex.runners.ingestion.stream_ingestion",
    # Dispatcher runner
    "invoke_dispatcher": "contextunity.router.cortex.runners.dispatcher.invoke_dispatcher",
    "stream_dispatcher": "contextunity.router.cortex.runners.dispatcher.stream_dispatcher",
}


def __getattr__(name: str) -> Any:
    """Lazy re-exports to avoid import-time side effects and circular imports."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    path = _EXPORTS[name]
    mod_name, attr = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)
