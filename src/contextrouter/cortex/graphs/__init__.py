"""
ContextRouter graphs package.

Structure:
    graphs/
    ├── dispatcher.py         # Central graph selection (by config/registry)
    │
    ├── rag_retrieval/        # RAG pipeline (retrieve → generate)
    │   ├── graph.py          # Main graph definition
    │   ├── extract.py        # Query extraction
    │   ├── intent.py         # Intent detection
    │   ├── retrieve.py       # Document retrieval
    │   ├── generate.py       # Response generation
    │   └── suggest.py        # Search suggestions
    │
    ├── contextmed/           # Medical analytics agent
    │   ├── graph.py          # LangGraph (planner → sql → verifier → viz)
    │   └── prompts.py        # System prompts
    │
    └── commerce/             # Commerce domain (subgraph architecture)
        ├── graph.py          # CommerceGraph (programmatic entry)
        ├── queue/            # Redis enrichment queue
        ├── ontology/         # KG relation definitions
        ├── chat/             # LLM intent detection (user messages)
        ├── gardener/         # Taxonomy enrichment
        ├── lexicon/          # Content generation
        └── matcher/          # Product matching

Usage:
    # Via dispatcher (config-based)
    from contextrouter.cortex.graphs import compile_graph
    graph = compile_graph()  # Uses router.graph config

    # Direct access (no heavy imports)
    from contextrouter.cortex.graphs.contextmed.graph import build_contextmed_graph
    from contextrouter.cortex.graphs.commerce import build_commerce_graph
"""

from __future__ import annotations

import importlib
from typing import Any

# ── Lazy imports to avoid pulling in heavy dependencies (joblib, etc.)
# when only a lightweight subgraph is needed. ──


def __getattr__(name: str) -> Any:
    """Lazy-load graph submodules on first access."""
    # Commerce graph exports
    _commerce_exports = {
        "ChatState",
        "CommerceState",
        "GardenerState",
        "MatcherState",
        "MatchingNode",
        "build_commerce_graph",
        "create_chat_subgraph",
        "create_gardener_subgraph",
        "create_lexicon_subgraph",
        "create_matcher_subgraph",
        "invoke_chat",
    }
    if name in _commerce_exports:
        mod = importlib.import_module(".commerce", __name__)
        return getattr(mod, name)

    # RAG graph
    if name == "rag_retrieval":
        return importlib.import_module(".rag_retrieval.graph", __name__)

    # Dispatcher
    _dispatcher_exports = {"build_graph", "compile_graph", "reset_graph"}
    if name in _dispatcher_exports:
        mod = importlib.import_module(".dispatcher", __name__)
        return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Dispatcher
    "build_graph",
    "compile_graph",
    "reset_graph",
    # Commerce
    "build_commerce_graph",
    "CommerceState",
    # Chat
    "create_chat_subgraph",
    "invoke_chat",
    "ChatState",
    # Gardener
    "create_gardener_subgraph",
    "GardenerState",
    # Lexicon
    "create_lexicon_subgraph",
    # Matcher
    "create_matcher_subgraph",
    "MatcherState",
    "MatchingNode",
    # RAG
    "rag_retrieval",
]
