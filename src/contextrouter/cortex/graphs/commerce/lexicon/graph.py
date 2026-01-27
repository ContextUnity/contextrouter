"""
Lexicon subgraph definition.

Lexicon: AI content generation for products using Perplexity.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# TODO: Move from lexicon.py when ready
# For now, placeholder implementation


class LexiconState:
    """State for Lexicon subgraph."""
    pass


async def generate_content_node(state: dict) -> dict:
    """Generate AI content for products."""
    logger.info("Lexicon content generation placeholder")
    return {"result": {"message": "Lexicon not yet migrated to subgraph"}}


def create_lexicon_subgraph():
    """Create Lexicon subgraph for AI content generation.

    TODO: Migrate from lexicon.py
    """
    graph = StateGraph(dict)

    graph.add_node("generate", generate_content_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)

    return graph.compile()
