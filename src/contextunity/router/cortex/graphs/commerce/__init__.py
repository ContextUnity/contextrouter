"""
Commerce graph package.

Entry point for all commerce-related AI operations.

Structure:
    commerce/
    ├── graph.py          # CommerceGraph - main entry (programmatic)
    ├── chat/             # Chat subgraph (LLM intent detection)
    ├── gardener/         # Taxonomy enrichment
    ├── lexicon/          # Content generation
    └── matcher/          # Product matching
"""

from .chat import ChatState, create_chat_subgraph, invoke_chat
from .gardener import GardenerState, build_gardener_graph
from .graph import build_commerce_graph
from .lexicon import create_lexicon_subgraph
from .matcher import MatcherState, create_matcher_subgraph, create_rlm_bulk_matcher_subgraph
from .state import CommerceState
from .writer.graph import build_writer_graph

__all__ = [
    # Main graph
    "build_commerce_graph",
    "CommerceState",
    # Chat (LLM intent detection)
    "create_chat_subgraph",
    "invoke_chat",
    "ChatState",
    # Gardener
    "build_gardener_graph",
    "GardenerState",
    # Lexicon
    "create_lexicon_subgraph",
    # Matcher
    "create_matcher_subgraph",
    "create_rlm_bulk_matcher_subgraph",
    "MatcherState",
    # Writer
    "build_writer_graph",
]
