"""
Matcher Agent - LangGraph implementation (STUB).

The Linker: Connects supplier products to canonical products.

Flow:
1. fetch_unmatched - Get unmatched supplier products
2. analyze - Vectorize, extract NER
3. search_canonical - Search in Oscar DB + Brain
4. score_candidates - LLM ranking
5. link_or_queue - Auto-link (high confidence) or add to queue
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


class MatcherState(TypedDict):
    """State for Matcher graph."""

    # Input
    batch_size: int
    confidence_threshold: float

    # Processing
    unmatched_products: List[Dict[str, Any]]
    analyzed: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]

    # Output
    auto_linked: int
    queued_for_review: int

    # Metrics
    total_tokens: int
    errors: List[str]


async def fetch_unmatched_node(state: MatcherState) -> MatcherState:
    """Fetch unmatched products from Commerce."""
    from ..tools.commerce_client import get_commerce_client

    client = get_commerce_client()
    products = await client.get_unmatched_products(limit=state["batch_size"])

    state["unmatched_products"] = products
    logger.info(f"Fetched {len(products)} unmatched products")
    return state


async def analyze_raw_node(state: MatcherState) -> MatcherState:
    """Analyze raw product data - NER, vectorize."""
    # TODO: Implement
    # - Extract brand, model, SKU patterns
    # - Generate embedding for name
    state["analyzed"] = state["unmatched_products"]
    return state


async def search_canonical_node(state: MatcherState) -> MatcherState:
    """Search for canonical products in Oscar + Brain."""

    # TODO: Implement
    # - Vector search in Brain
    # - Exact SKU match in Oscar
    # - Fuzzy name matching

    state["candidates"] = []
    return state


async def score_candidates_node(state: MatcherState) -> MatcherState:
    """Score candidates using LLM."""
    # TODO: Implement
    # - LLM ranks candidates
    # - Returns confidence scores
    return state


async def link_or_queue_node(state: MatcherState) -> MatcherState:
    """Link high-confidence matches, queue others."""
    from ..tools.commerce_client import get_commerce_client

    client = get_commerce_client()
    linked = 0
    queued = 0

    for candidate in state.get("candidates", []):
        if candidate.get("confidence", 0) >= state["confidence_threshold"]:
            await client.link_product(
                candidate["supplier_id"], candidate["canonical_id"], candidate["confidence"]
            )
            linked += 1
        else:
            await client.add_to_unmatched_queue(
                candidate["supplier_id"], candidate.get("alternatives", []), "Low confidence"
            )
            queued += 1

    state["auto_linked"] = linked
    state["queued_for_review"] = queued
    return state


def create_matcher_graph() -> StateGraph:
    """Create Matcher LangGraph."""
    graph = StateGraph(MatcherState)

    graph.add_node("fetch_unmatched", fetch_unmatched_node)
    graph.add_node("analyze", analyze_raw_node)
    graph.add_node("search_canonical", search_canonical_node)
    graph.add_node("score_candidates", score_candidates_node)
    graph.add_node("link_or_queue", link_or_queue_node)

    graph.add_edge(START, "fetch_unmatched")
    graph.add_edge("fetch_unmatched", "analyze")
    graph.add_edge("analyze", "search_canonical")
    graph.add_edge("search_canonical", "score_candidates")
    graph.add_edge("score_candidates", "link_or_queue")
    graph.add_edge("link_or_queue", END)

    return graph.compile()


async def invoke_matcher(
    batch_size: int = 100, confidence_threshold: float = 0.95
) -> Dict[str, Any]:
    """Run Matcher agent."""
    graph = create_matcher_graph()

    initial_state: MatcherState = {
        "batch_size": batch_size,
        "confidence_threshold": confidence_threshold,
        "unmatched_products": [],
        "analyzed": [],
        "candidates": [],
        "auto_linked": 0,
        "queued_for_review": 0,
        "total_tokens": 0,
        "errors": [],
    }

    final_state = await graph.ainvoke(initial_state)

    return {
        "processed": len(final_state["unmatched_products"]),
        "auto_linked": final_state["auto_linked"],
        "queued": final_state["queued_for_review"],
    }
