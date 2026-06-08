"""RAG cortex - LangGraph agent implementation.

This package provides the LangGraph-based RAG agent for knowledge-driven question answering.

Usage:
    from contextunity.router.cortex import compile_graph
    from contextunity.router.modules.observability import get_langfuse_callbacks

    # Compile and invoke with tracing
    graph = compile_graph()
    callbacks = get_langfuse_callbacks(session_id="my_session", user_id="user123")
    result = graph.invoke(input_state, config={"callbacks": callbacks})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextunity.router.cortex.builder import build_graph, compile_graph, reset_graph
    from contextunity.router.cortex.core_state import (
        get_last_user_query,
    )
    from contextunity.router.cortex.dispatcher import invoke_dispatcher, stream_dispatcher
    from contextunity.router.cortex.services import get_graph_service
    from contextunity.router.modules.retrieval.rag.models import Citation, RetrievedDoc

__all__ = [
    "get_last_user_query",
    "Citation",
    "RetrievedDoc",
    "get_graph_service",
    "build_graph",
    "compile_graph",
    "reset_graph",
    # Dispatcher runner
    "invoke_dispatcher",
    "stream_dispatcher",
]


def __getattr__(name: str) -> object:
    """Lazy re-exports to avoid import-time side effects and circular imports."""
    if name == "get_last_user_query":
        from contextunity.router.cortex.core_state import get_last_user_query

        return get_last_user_query
    if name == "Citation":
        from contextunity.router.modules.retrieval.rag.models import Citation

        return Citation
    if name == "RetrievedDoc":
        from contextunity.router.modules.retrieval.rag.models import RetrievedDoc

        return RetrievedDoc
    if name == "get_graph_service":
        from contextunity.router.cortex.services import get_graph_service

        return get_graph_service
    if name == "build_graph":
        from contextunity.router.cortex.builder import build_graph

        return build_graph
    if name == "compile_graph":
        from contextunity.router.cortex.builder import compile_graph

        return compile_graph
    if name == "reset_graph":
        from contextunity.router.cortex.builder import reset_graph

        return reset_graph
    if name == "invoke_dispatcher":
        from contextunity.router.cortex.dispatcher import invoke_dispatcher

        return invoke_dispatcher
    if name == "stream_dispatcher":
        from contextunity.router.cortex.dispatcher import stream_dispatcher

        return stream_dispatcher
    raise AttributeError(name)
