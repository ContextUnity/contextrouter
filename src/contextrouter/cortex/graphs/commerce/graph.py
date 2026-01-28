"""
Commerce graph - main entry point for commerce AI operations.

Routes to subgraphs based on intent:
- enrich → Gardener subgraph (taxonomy, NER, KG)
- generate_content → Lexicon subgraph (AI content generation)
- match_products → Matcher subgraph (product deduplication)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from contextrouter.core.registry import register_graph

from .gardener import create_gardener_subgraph
from .lexicon import create_lexicon_subgraph
from .matcher import create_matcher_subgraph
from .state import CommerceState

logger = logging.getLogger(__name__)


async def route_intent_node(state: CommerceState) -> Dict[str, Any]:
    """Route to appropriate subgraph based on intent."""
    intent = state.get("intent", "search")
    logger.info(f"Commerce routing intent: {intent}")
    return {"intent": intent}


def _get_next_node(state: CommerceState) -> str:
    """Determine next node based on intent."""
    intent = state.get("intent", "search")
    routing = {
        "enrich": "gardener",
        "generate_content": "lexicon",
        "match_products": "matcher",
    }
    return routing.get(intent, END)


async def search_node(state: CommerceState) -> Dict[str, Any]:
    """Placeholder for product search (routes to Brain retrieval)."""
    return {"result": {"message": "Search not implemented, use Brain retrieval"}}


@register_graph("commerce")
def build_commerce_graph():
    """Build Commerce graph with subgraphs.

    Usage:
        from contextrouter.cortex.graphs.commerce import build_commerce_graph

        graph = build_commerce_graph()
        result = await graph.ainvoke({
            "intent": "enrich",
            "tenant_id": "default",
            "db_url": "postgresql://...",
            "batch_size": 10,
            "prompts_dir": "/path/to/prompts",
        })
    """
    workflow = StateGraph(CommerceState)

    # Router node
    workflow.add_node("route", route_intent_node)

    # Subgraphs
    workflow.add_node("gardener", create_gardener_subgraph())
    workflow.add_node("lexicon", create_lexicon_subgraph())
    workflow.add_node("matcher", create_matcher_subgraph())

    # Placeholder nodes
    workflow.add_node("search", search_node)

    # Entry
    workflow.set_entry_point("route")

    # Conditional routing
    workflow.add_conditional_edges(
        "route",
        _get_next_node,
        {
            "gardener": "gardener",
            "lexicon": "lexicon",
            "matcher": "matcher",
            END: END,
        },
    )

    # All subgraphs exit to END
    workflow.add_edge("gardener", END)
    workflow.add_edge("lexicon", END)
    workflow.add_edge("matcher", END)
    workflow.add_edge("search", END)

    return workflow.compile()
