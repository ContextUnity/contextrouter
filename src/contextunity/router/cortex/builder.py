"""Graph Dispatcher — Centralized Selection and Registry for LangGraph Topologies.

This module acts as the core dispatcher for resolved LangGraph instances. It abstracts the selection,
dynamic loading, template compilation, and instantiation of StateGraph configurations based on project
specifications and tenant environment settings.

Resolution Priority Logic:
    1. Explicit `graph_name` argument (allows on-the-fly invocation of specific graphs).
    2. Dynamic override path via configuration (`router.override_path` specifying a Python attribute path).
    3. Registered user/extension graph mappings retrieved from the global `graph_registry`.
    4. Built-in template keys (such as `rag_retrieval`, `gardener`, `enricher`, or `dispatcher`).
    5. Fallback Default: `rag_retrieval` (mapped internally to the `retrieval_augmented` template).

Dynamic Override Protocol:
    Power users and tenant services can override default execution paths by configuring a custom
    object locator at `router.override_path` (e.g., `myapp.graphs:my_custom_builder`). The loader:
        - Resolves and imports the target module.
        - Obtains the specified attribute object.
        - Executes either the `build_graph()` method on the object or the callable itself.
        - Verifies that the returned value strictly inherits from `langgraph.graph.StateGraph`.

Caching Lifecycle:
    To minimize graph compilation overhead during active gRPC stream execution, the default compiled
    state graph is cached at the module level in `_compiled_graph`. Testing suites, dynamic schema reloads,
    and manifest pushes call `reset_graph()` to invalidate this cache and trigger recompilation.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeGuard, runtime_checkable

if TYPE_CHECKING:
    from contextunity.router.cortex.dispatcher_agent.types import DispatcherState

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from contextunity.router.core import get_core_config
from contextunity.router.core.exceptions import RouterGraphBuilderError
from contextunity.router.core.registry import graph_registry
from contextunity.router.cortex.types import GraphState, RunnableGraphFactory

# Type aliases — the builder factory can produce either state variant.
_DefaultGraph: TypeAlias = StateGraph[GraphState, None, GraphState, GraphState]
_DispatcherGraph: TypeAlias = "StateGraph[DispatcherState, None, DispatcherState, DispatcherState]"
_AnyGraph: TypeAlias = _DefaultGraph | _DispatcherGraph

_DefaultCompiled: TypeAlias = CompiledStateGraph[GraphState, None, GraphState, GraphState]
_DispatcherCompiled: TypeAlias = (
    "CompiledStateGraph[DispatcherState, None, DispatcherState, DispatcherState]"
)
_AnyCompiled: TypeAlias = _DefaultCompiled | _DispatcherCompiled


@runtime_checkable
class _OverrideStateGraphFactory(Protocol):
    """Legacy ``router.override_path`` objects that expose ``build_graph()`` (not registry ``build()``)."""

    def build_graph(self) -> _DefaultGraph: ...


@runtime_checkable
class _CompilableWorkflow(Protocol):
    """Structural compile surface for built graph topologies."""

    def compile(self) -> _AnyCompiled: ...


def _is_any_graph(value: object) -> TypeGuard[_AnyGraph]:
    """Narrow dynamic graph products to the builder return union."""
    return isinstance(value, StateGraph)


def _require_uncompiled_graph(value: object, context: str) -> _AnyGraph:
    """Validate override/registry products as uncompiled LangGraph topologies."""
    if not _is_any_graph(value):
        raise RouterGraphBuilderError(f"{context} returned invalid type: {type(value)}")
    return value


def _compile_workflow(workflow: _CompilableWorkflow) -> _AnyCompiled:
    """Compile through a structural boundary instead of vendor overloads."""
    return workflow.compile()


_compiled_graph: _AnyCompiled | None = None


def build_graph(
    graph_name: str | None = None,
) -> _AnyGraph:
    """Resolve, load, and build an uncompiled StateGraph configuration.

    This function performs structural lookup of the requested topology name. It merges
    custom plugin registries with built-in declarative templates to yield a validated,
    uncompiled LangGraph.

    Args:
        graph_name: Optional explicit name of the graph to instantiate. If omitted,
            the active configuration settings in `router.graph` or `router.override_path`
            determine the lookup target.

    Returns:
        StateGraph: The uncompiled LangGraph topology ready for nodes and compilation.

    Raises:
        RouterGraphBuilderError: If the resolved path is invalid, the target function is not
            callable, or the built object does not conform to the `StateGraph` interface,
            or if an unknown template key is requested.
    """
    from .compiler.builder import build_from_template

    cfg = get_core_config()

    # Explicit override path still wins (power-user wiring)
    if not graph_name and cfg.router.override_path:
        raw = (cfg.router.override_path or "").strip()
        if not raw:
            raise RouterGraphBuilderError("Empty router.override_path")
        if ":" in raw:
            mod_name, attr = raw.split(":", 1)
        else:
            mod_name, attr = raw.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        raw_obj: object = getattr(mod, attr, None)
        if raw_obj is None:
            raise RouterGraphBuilderError(f"Missing attribute {attr!r} in module {mod_name!r}")
        if isinstance(raw_obj, RunnableGraphFactory):
            return _require_uncompiled_graph(raw_obj.build(), f"{raw}.build()")

        if isinstance(raw_obj, _OverrideStateGraphFactory):
            return raw_obj.build_graph()

        if callable(raw_obj):
            return _require_uncompiled_graph(raw_obj(), f"Callable {raw}")

        raise RouterGraphBuilderError(f"router.override_path object is not callable: {raw}")

    # Determine graph key
    key = graph_name or (cfg.router.graph or "rag_retrieval").strip() or "rag_retrieval"

    # Check registry first (user-registered graphs)
    if graph_registry.has(key):
        return _require_uncompiled_graph(
            graph_registry.get(key).build(),
            f"Registered graph '{key}'",
        )

    # Built-in graphs
    from .dispatcher_agent import build_dispatcher_graph

    # Dispatcher uses DispatcherState (extends GraphState) — handle separately
    # because StateGraph type params are invariant.
    if key == "dispatcher":
        return build_dispatcher_graph()

    builtin_graphs = {
        "rag_retrieval": lambda: build_from_template("retrieval_augmented"),
        "sql_analytics": lambda: build_from_template("retrieval_augmented"),
        # Domain templates (compose universal router_* platform tools)
        "gardener": lambda: build_from_template("gardener"),
        "enricher": lambda: build_from_template("enricher"),
        "news_pipeline": lambda: build_from_template("news_pipeline"),
        "rlm_bulk_matcher": lambda: build_from_template("rlm_bulk_matcher"),
    }

    if key not in builtin_graphs:
        known = sorted(
            set(graph_registry.list_keys()) | set(builtin_graphs.keys()) | {"dispatcher"}
        )
        raise RouterGraphBuilderError(f"Unknown graph='{key}'. Known: {known}")

    return builtin_graphs[key]()


def compile_graph(
    graph_name: str | None = None,
) -> _AnyCompiled:
    """Compile the resolved StateGraph and cache the compiled result.

    Compiles the target topology into a runtime-ready `CompiledStateGraph` capable of
    asynchronous state transitions (`ainvoke`/`astream`).

    Note:
        Explicitly passing `graph_name` forces a direct compilation cycle and bypasses
        the shared module-level cache. Calling `compile_graph(None)` returns the
        globally cached default instance, initializing it on the first execution.

    Args:
        graph_name: Optional explicit name of the graph to compile.

    Returns:
        CompiledStateGraph: A compiled LangGraph workspace ready to execute.

    Raises:
        RouterGraphBuilderError: If the selected configuration fails compilation.
    """
    global _compiled_graph

    # If specific name requested, don't use cache
    if graph_name:
        workflow = build_graph(graph_name)
        return _compile_workflow(workflow)

    if _compiled_graph is None:
        _compiled_graph = _compile_workflow(build_graph())

    return _compiled_graph


def reset_graph() -> None:
    """Reset the module-level compiled graph cache.

    Forces a full recompilation cycle on the next call to `compile_graph()`. Typically used
    during testing, configuration dynamic reloads, or environment manifest updates.
    """
    global _compiled_graph
    _compiled_graph = None


__all__ = ["build_graph", "compile_graph", "reset_graph"]
