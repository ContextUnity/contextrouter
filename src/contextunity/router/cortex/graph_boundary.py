"""LangGraph vendor boundary — narrow Protocol surfaces for graph construction."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Protocol

from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver

from .types import GraphState, NodeFunc, RunnableGraph, is_runnable_graph


class _GraphNodeAdder(Protocol):
    """LangGraph ``add_node`` surface without vendor overload leakage."""

    def add_node(
        self,
        node: str,
        action: NodeFunc | Runnable[object, object],
    ) -> object:
        """Register a node action on the graph."""
        ...


class _GraphConditionalRouter(Protocol):
    """LangGraph conditional edge surface with a narrow path map."""

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[[GraphState], Hashable | Sequence[Hashable]],
        path_map: dict[Hashable, str] | None = None,
    ) -> object:
        """Register conditional routing for *source*."""
        ...


class _GraphCompiler(Protocol):
    """LangGraph compile surface with optional checkpointing."""

    def compile(
        self,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> object:
        """Compile the graph into a runnable product."""
        ...


def graph_add_typed_node(
    graph: _GraphNodeAdder,
    node: str,
    action: NodeFunc | Runnable[object, object],
) -> None:
    """Add a named node through the vendor boundary."""
    _ = graph.add_node(node, action)


def graph_add_typed_conditional_edges(
    graph: _GraphConditionalRouter,
    source: str,
    path: Callable[[GraphState], Hashable | Sequence[Hashable]],
    path_map: Mapping[str, str],
) -> None:
    """Add conditional routing through the vendor boundary."""
    hash_path_map: dict[Hashable, str] = {}
    for route_key, route_target in path_map.items():
        hash_path_map[route_key] = route_target
    _ = graph.add_conditional_edges(source, path, hash_path_map)


def graph_compile_typed(
    graph: _GraphCompiler,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> RunnableGraph:
    """Compile a graph through the vendor boundary."""
    compiled: object = (
        graph.compile() if checkpointer is None else graph.compile(checkpointer=checkpointer)
    )
    if not is_runnable_graph(compiled):
        msg = f"Unexpected compile product type: {type(compiled).__name__}"
        raise TypeError(msg)
    return compiled
