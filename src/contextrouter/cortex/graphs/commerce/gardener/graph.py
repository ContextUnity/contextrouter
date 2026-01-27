"""
Gardener subgraph definition.

Gardener: Enriches products with taxonomy, NER, parameters, technologies, KG.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    classify_taxonomy_node,
    extract_ner_node,
    fetch_pending_node,
    update_kg_node,
    write_results_node,
)
from .state import GardenerState


def create_gardener_subgraph() -> StateGraph:
    """Create Gardener subgraph for taxonomy classification.

    This subgraph is embedded in the Commerce graph when intent="enrich".

    Flow:
        fetch_pending → classify_taxonomy → extract_ner → update_kg → write_results
    """
    graph = StateGraph(GardenerState)

    graph.add_node("fetch_pending", fetch_pending_node)
    graph.add_node("classify_taxonomy", classify_taxonomy_node)
    graph.add_node("extract_ner", extract_ner_node)
    graph.add_node("update_kg", update_kg_node)
    graph.add_node("write_results", write_results_node)

    graph.add_edge(START, "fetch_pending")
    graph.add_edge("fetch_pending", "classify_taxonomy")
    graph.add_edge("classify_taxonomy", "extract_ner")
    graph.add_edge("extract_ner", "update_kg")
    graph.add_edge("update_kg", "write_results")
    graph.add_edge("write_results", END)

    return graph.compile()
