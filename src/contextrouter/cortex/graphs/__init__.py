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

    # Direct access
    from contextrouter.cortex.graphs.commerce import build_commerce_graph
    commerce = build_commerce_graph()

    # RAG graph
    from contextrouter.cortex.graphs.rag_retrieval import compile_graph
    rag = compile_graph()
"""

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
from .rag_retrieval import graph as rag_retrieval

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
