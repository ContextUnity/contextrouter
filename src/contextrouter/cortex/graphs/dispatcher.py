"""
Graph Dispatcher - Central graph selection and registry.

This module dispatches to the correct graph based on configuration.
Graphs can be:
1. Registered via @register_graph decorator
2. Overridden via router.override_path config
3. Built-in (rag_retrieval, commerce)

Usage:
    # Get compiled graph based on config
    graph = compile_graph()
    result = await graph.ainvoke({...})

    # Or build specific graph directly
    from .commerce import build_commerce_graph
    commerce = build_commerce_graph()

Configuration:
    # In config or environment
    router.graph = "commerce"      # Use commerce graph
    router.graph = "rag_retrieval" # Use RAG graph (default)
    router.override_path = "myapp.graphs:custom_builder"  # Custom graph
"""

from __future__ import annotations

import importlib
from typing import Callable, cast

from contextrouter.core import get_core_config
from contextrouter.core.registry import graph_registry

_compiled_graph: object | None = None


def build_graph(graph_name: str | None = None):
    """Build graph by name or from config.

    Priority:
    1. Explicit graph_name argument
    2. router.override_path config (custom Python path)
    3. router.graph config (registered or built-in name)
    4. Default: rag_retrieval

    Args:
        graph_name: Optional explicit graph name to build

    Returns:
        Uncompiled StateGraph
    """
    from .commerce import build_commerce_graph
    from .rag_retrieval.graph import build_graph as build_rag_graph

    cfg = get_core_config()

    # Explicit override path still wins (power-user wiring)
    if not graph_name and cfg.router.override_path:
        raw = (cfg.router.override_path or "").strip()
        if not raw:
            raise ValueError("Empty router.override_path")
        if ":" in raw:
            mod_name, attr = raw.split(":", 1)
        else:
            mod_name, attr = raw.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        if hasattr(obj, "build_graph"):
            return obj.build_graph()
        if callable(obj):
            return obj()
        raise TypeError(f"router.override_path object is not callable: {raw}")

    # Determine graph key
    key = graph_name or (cfg.router.graph or "rag_retrieval").strip() or "rag_retrieval"

    # Check registry first (user-registered graphs)
    if graph_registry.has(key):
        return graph_registry.get(key)()

    # Built-in graphs
    from .dispatcher_agent import build_dispatcher_graph

    builtin_graphs: dict[str, Callable[[], object]] = {
        "rag_retrieval": build_rag_graph,
        "commerce": build_commerce_graph,
        "dispatcher": build_dispatcher_graph,
    }

    # Optional: ContextZero privacy proxy graph
    try:
        from contextzero.graph import build_zero_graph

        builtin_graphs["privacy_proxy"] = build_zero_graph
    except ImportError:
        pass

    if key not in builtin_graphs:
        known = sorted(set(graph_registry.list_keys()) | set(builtin_graphs.keys()))
        raise KeyError(f"Unknown graph='{key}'. Known: {known}")

    return builtin_graphs[key]()


def compile_graph(graph_name: str | None = None) -> object:
    """Compile and return graph (cached).

    Args:
        graph_name: Optional explicit graph name. If None, uses config.

    Returns:
        Compiled LangGraph ready for invoke/ainvoke.
    """
    global _compiled_graph

    # If specific name requested, don't use cache
    if graph_name:
        workflow = build_graph(graph_name)
        return workflow.compile()

    # Default: use cached graph from config
    if _compiled_graph is None:
        workflow = build_graph()
        _compiled_graph = workflow.compile()
    return cast(object, _compiled_graph)


def reset_graph() -> None:
    """Reset cached graph (for testing or config reload)."""
    global _compiled_graph
    _compiled_graph = None


__all__ = ["build_graph", "compile_graph", "reset_graph"]
