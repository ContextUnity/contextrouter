"""
ContextRouter graphs package.

Structure:
    graphs/
    ├── dispatcher.py         # Central graph selection (by config/registry)
    ├── rag_retrieval.py      # RAG pipeline (retrieve → generate)
    │
    └── commerce/             # Commerce domain (subgraph architecture)
        ├── graph.py          # CommerceGraph (programmatic entry)
        ├── chat/             # LLM intent detection (user messages)
        ├── gardener/         # Taxonomy enrichment
        ├── lexicon/          # Content generation
        └── matcher/          # Product matching

Usage:
    # Via dispatcher (config-based)
    from contextrouter.cortex.graphs import compile_graph
    graph = compile_graph()  # Uses router.graph config

    # Direct access
    from contextrouter.cortex.graphs.commerce import build_commerce_graph
    commerce = build_commerce_graph()
"""

from . import rag_retrieval

# Commerce graph
from .commerce import (
    ChatState,
    CommerceState,
    GardenerState,
    MatcherState,
    MatchingNode,
    build_commerce_graph,
    create_chat_subgraph,
    create_gardener_subgraph,
    create_lexicon_subgraph,
    create_matcher_subgraph,
    invoke_chat,
)

# Dispatcher (central graph selection)
from .dispatcher import build_graph, compile_graph, reset_graph

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
