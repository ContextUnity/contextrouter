"""LangGraph graph definition for the RAG agent.

This module defines and compiles the LangGraph StateGraph for the RAG agent.
The graph implements a retrieval-augmented generation pipeline with intent
detection, parallel retrieval, and parallel suggestions + generation.

Graph Flow:
    START -> extract_query -> detect_intent -> [conditional]
                                                    |
                              +---------+-----------+
                              |                     |
                          retrieve              generate (skip retrieval)
                              |                     |
                    +---------+---------+           |
                    |                   |           |
                suggest             generate        |
                    |                   |           |
                   END                 END         END

Nodes:
    - extract_query: Extracts the user query from the last HumanMessage.
    - detect_intent: Classifies intent and derives retrieval queries.
    - retrieve: Parallel retrieval from Vertex AI Search + optional web search.
    - suggest: Generates follow-up search suggestions (runs in parallel with generate).
    - generate: Produces the final assistant response (runs in parallel with suggest).

Conditional Routing:
    - If intent is 'rag_and_web' and query exists -> retrieve -> [suggest || generate] -> END
    - Otherwise (translate/summarize/rewrite) -> [suggest || generate] -> END
"""

from __future__ import annotations

import importlib
from typing import cast

from langgraph.graph import END, START, StateGraph

import contextrouter.core.registry as core_registry_module
from contextrouter.core import agent_registry, get_core_config
from contextrouter.cortex import AgentState, InputState, OutputState

from ..steps.rag_retrieval.routing import should_retrieve

_compiled_graph: object | None = None


def _import_object(path: str) -> object:
    """Import an object from a dotted path.

    Supports both 'pkg.mod:Attr' and 'pkg.mod.Attr'.
    """

    raw = (path or "").strip()
    if not raw:
        raise ValueError("Empty import path")
    if ":" in raw:
        mod_name, attr = raw.split(":", 1)
    elif "." in raw:
        mod_name, attr = raw.rsplit(".", 1)
    else:
        # Simple module name; assume the object has the same name as the module.
        mod_name = raw
        attr = raw
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def build_graph() -> StateGraph:
    """Build the RAG agent graph (not compiled).

    Creates a StateGraph with the following structure:
    - extract_query -> detect_intent -> conditional routing
    - If should_retrieve=True: retrieve -> [suggest || generate] -> END
    - If should_retrieve=False: [suggest || generate] -> END

    Suggest and generate run in parallel after retrieval completes.

    Returns:
        Uncompiled StateGraph ready for compilation.
    """
    # Router override (composition): host may supply a custom graph builder.
    override_path = get_core_config().router.override_path
    if override_path:
        builder = _import_object(override_path)
        if hasattr(builder, "build_graph"):
            return builder.build_graph()  # type: ignore[no-any-return]
        if callable(builder):
            return builder()  # type: ignore[no-any-return]
        raise TypeError(f"router.override_path object is not callable: {override_path}")

    workflow = StateGraph(AgentState, input=InputState, output=OutputState)

    mode = (get_core_config().router.mode or "agent").strip().lower()
    if mode not in {"agent", "direct"}:
        mode = "agent"

    if mode == "direct":
        # Direct-mode: assemble graph from function nodes.
        from contextrouter.cortex.steps.rag_retrieval import (
            detect_intent,
            extract_user_query,
            generate_response,
            generate_search_suggestions,
            retrieve_documents,
        )

        workflow.add_node("extract_query", extract_user_query)
        workflow.add_node("detect_intent", detect_intent)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("suggest", generate_search_suggestions)
        workflow.add_node("generate", generate_response)
    else:
        # Agent-mode: assemble graph dynamically from registry (class-based nodes).
        extract_cls = agent_registry.get("extract_query")
        intent_cls = agent_registry.get("detect_intent")
        retrieve_cls = agent_registry.get("retrieve")
        suggest_cls = agent_registry.get("suggest")
        generate_cls = agent_registry.get("generate")

        workflow.add_node("extract_query", extract_cls(core_registry_module))
        workflow.add_node("detect_intent", intent_cls(core_registry_module))
        workflow.add_node("retrieve", retrieve_cls(core_registry_module))
        workflow.add_node("suggest", suggest_cls(core_registry_module))
        workflow.add_node("generate", generate_cls(core_registry_module))

    workflow.add_edge(START, "extract_query")
    workflow.add_edge("extract_query", "detect_intent")

    workflow.add_conditional_edges(
        "detect_intent",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "skip_retrieve": "generate",
        },
    )

    # After retrieve, fan out to both suggest and generate in parallel
    workflow.add_edge("retrieve", "suggest")
    workflow.add_edge("retrieve", "generate")

    # Suggest goes to END (runs in parallel with generate after retrieve)
    workflow.add_edge("suggest", END)

    # Generate converges to END
    workflow.add_edge("generate", END)

    return workflow


def compile_graph() -> object:
    """Compile and return the RAG agent graph.

    Returns a compiled LangGraph that can be invoked with:
        result = graph.invoke({"messages": [...], "session_id": "...", "platform": "..."})

    Or streamed with:
        async for event in graph.astream_events(input, version="v2"):
            ...
    """
    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_graph()
        _compiled_graph = workflow.compile()
    return cast(object, _compiled_graph)


def reset_graph() -> None:
    """Reset compiled graph (for testing)."""
    global _compiled_graph
    _compiled_graph = None
