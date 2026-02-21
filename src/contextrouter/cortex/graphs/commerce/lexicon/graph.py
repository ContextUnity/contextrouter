"""
Lexicon subgraph definition.

Lexicon: AI content generation for products using LLM.

Flow:
    analyze → generate → validate → write_results

Embedded in the Commerce graph when intent="generate_content".
Also callable standalone via `create_lexicon_subgraph()`.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .nodes import (
    analyze_products_node,
    generate_content_node,
    validate_content_node,
    write_results_node,
)
from .state import LexiconState

logger = logging.getLogger(__name__)


def create_lexicon_subgraph():
    """Create Lexicon subgraph for AI content generation.

    This subgraph is embedded in the Commerce graph when intent="generate_content".

    Flow:
        analyze → generate → validate → write_results → END

    Usage:
        graph = create_lexicon_subgraph()
        result = await graph.ainvoke({
            "tenant_id": "default",
            "brain_url": "localhost:50051",
            "product_ids": [1, 2, 3],
            "language": "uk",
        })
    """
    graph = StateGraph(LexiconState)

    graph.add_node("analyze", analyze_products_node)
    graph.add_node("generate", generate_content_node)
    graph.add_node("validate", validate_content_node)
    graph.add_node("write_results", write_results_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "write_results")
    graph.add_edge("write_results", END)

    return graph.compile()


# Alias for runner compatibility
compile_lexicon_graph = create_lexicon_subgraph


__all__ = ["create_lexicon_subgraph", "compile_lexicon_graph"]
