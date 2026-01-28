"""
Matcher subgraph definition.

Matcher: Product deduplication and identity matching.
Supports both:
- Regular incremental matching (small batches)
- RLM bulk matching (50k+ products in single pass)
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


class RLMBulkMatcherState(TypedDict):
    """State for RLM Bulk Matcher subgraph.

    Designed for large-scale matching (50k supplier → 10k site).
    """

    # Config
    tenant_id: str
    confidence_threshold: float
    rlm_environment: str  # local, docker, modal, prime

    # Input products (can be very large)
    supplier_products: List[Dict[str, Any]]
    site_products: List[Dict[str, Any]]

    # Results
    matches: List[Dict[str, Any]]
    match_stats: Dict[str, Any]
    unmatched: List[Dict[str, Any]]


def create_matcher_subgraph():
    """Create Matcher subgraph for product deduplication.

    Flow:
        fetch_unmatched → match_products → link_or_queue

    Use this for regular incremental matching of small batches.
    For bulk matching (50k+), use create_rlm_bulk_matcher_subgraph().
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


def create_rlm_bulk_matcher_subgraph():
    """Create RLM Bulk Matcher subgraph for large-scale matching.

    Uses Recursive Language Models (RLM) for deep matching of
    50k+ supplier products against 10k site products in a single pass.

    Flow:
        rlm_bulk_match (single node - handles everything)

    Benefits:
    - 60-70% cost reduction vs chunked approach
    - 85%+ match rate with multi-factor comparison
    - Handles context degradation via REPL recursion

    Requires: pip install rlm (or uv add contextrouter[rlm])

    Usage:
        graph = create_rlm_bulk_matcher_subgraph()
        result = await graph.ainvoke({
            "supplier_products": supplier_list,  # 50k items
            "site_products": site_list,          # 10k items
            "confidence_threshold": 0.7,
            "rlm_environment": "docker",
        })
    """
    from .rlm_bulk import rlm_bulk_match_node

    graph = StateGraph(RLMBulkMatcherState)

    graph.add_node("rlm_bulk_match", rlm_bulk_match_node)

    graph.add_edge(START, "rlm_bulk_match")
    graph.add_edge("rlm_bulk_match", END)

    return graph.compile()
