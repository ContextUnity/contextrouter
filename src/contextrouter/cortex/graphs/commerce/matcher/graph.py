"""
Matcher subgraph definition.

Matcher: Product deduplication and identity matching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from .nodes import fetch_unmatched_node, link_or_queue_node, match_products_node

logger = logging.getLogger(__name__)


class MatcherState(TypedDict):
    """State for Matcher subgraph."""

    # Config
    batch_size: int
    db_url: str
    tenant_id: str

    # Products to match
    products: List[Dict[str, Any]]

    # Results
    match_results: List[Dict[str, Any]]
    auto_linked: int
    queued: int


def create_matcher_subgraph():
    """Create Matcher subgraph for product deduplication.

    Flow:
        fetch_unmatched → match_products → link_or_queue
    """
    graph = StateGraph(MatcherState)

    graph.add_node("fetch_unmatched", fetch_unmatched_node)
    graph.add_node("match_products", match_products_node)
    graph.add_node("link_or_queue", link_or_queue_node)

    graph.add_edge(START, "fetch_unmatched")
    graph.add_edge("fetch_unmatched", "match_products")
    graph.add_edge("match_products", "link_or_queue")
    graph.add_edge("link_or_queue", END)

    return graph.compile()
