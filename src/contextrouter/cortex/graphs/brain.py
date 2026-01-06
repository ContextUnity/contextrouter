"""Central brain graph router."""

from __future__ import annotations

import importlib
from typing import Callable, cast

from contextrouter.core import get_core_config
from contextrouter.core.registry import graph_registry
from contextrouter.cortex.graphs import rag_ingestion, rag_retrieval

_compiled_graph: object | None = None


def build_graph():
    """Build the central brain graph (not compiled)."""

    cfg = get_core_config()
    # Explicit override still wins (power-user wiring).
    if cfg.router.override_path:
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
            return obj.build_graph()  # type: ignore[no-any-return]
        if callable(obj):
            return obj()  # type: ignore[no-any-return]
        raise TypeError(f"router.override_path object is not callable: {raw}")

    # Check if a custom graph is registered
    key = (cfg.router.graph or "rag_retrieval").strip() or "rag_retrieval"
    if graph_registry.has(key):
        return graph_registry.get(key)()

    # Fallback to built-in graphs
    builtin_graphs: dict[str, Callable[[], object]] = {
        "rag_retrieval": rag_retrieval.build_graph,
        "rag_ingestion": rag_ingestion.build_graph,
    }

    if key not in builtin_graphs:
        known_graphs = sorted(set(graph_registry.list_keys()) | set(builtin_graphs.keys()))
        raise KeyError(f"Unknown router.graph='{key}'. Known: {known_graphs}")
    return builtin_graphs[key]()


def compile_graph() -> object:
    """Compile and return the central brain graph."""

    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_graph()
        _compiled_graph = workflow.compile()
    return cast(object, _compiled_graph)


def reset_graph() -> None:
    global _compiled_graph
    _compiled_graph = None


__all__ = ["build_graph", "compile_graph", "reset_graph"]
