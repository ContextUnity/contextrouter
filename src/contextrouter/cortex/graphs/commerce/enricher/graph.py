"""
Enricher graph builder.

Flow:
    init_credentials → normalize_raw → search_images → generate_description
        → ner_technologies → verify_technologies_bidi
        → create_missing_technology_articles (conditional) → map_attributes
"""

from __future__ import annotations

from typing import Any

from contextcore import get_context_unit_logger
from langgraph.graph import END, START, StateGraph

from .state import ProductEnricherState

logger = get_context_unit_logger(__name__)


def build_enricher_graph(config: dict) -> Any:
    """Build Enricher graph from registration config.

    Args:
        config: Graph configuration dict with keys:
            - model_key: Main LLM model for general parsing (e.g. 'openai/gpt-4o-mini')
            - perplexity_model: Model key for description generation (e.g. 'perplexity/sonar')
            - reasoning_effort: Main LLM reasoning effort
    """
    from .nodes import (
        make_create_missing_technology_articles,
        make_generate_description,
        make_init_credentials,
        make_map_attributes,
        make_ner_technologies,
        make_normalize_raw,
        make_search_images,
        make_verify_technologies_bidi,
        should_create_technologies,
    )

    model_key = config.get("model_key", "")
    perplexity_model = config.get("perplexity_model", "perplexity/sonar")

    if not model_key:
        logger.error("Enricher graph config missing 'model_key'")

    # Create nodes with config baked in (closure pattern)
    init_credentials = make_init_credentials()
    normalize_raw = make_normalize_raw()
    search_images = make_search_images()
    generate_description = make_generate_description()
    ner_technologies = make_ner_technologies()
    verify_technologies_bidi = make_verify_technologies_bidi()
    create_missing_technology_articles = make_create_missing_technology_articles()
    map_attributes = make_map_attributes()

    from contextrouter.cortex.graphs.secure_node import make_secure_node

    graph = StateGraph(ProductEnricherState)

    secure_init = make_secure_node("init_credentials", init_credentials)
    secure_normalize = make_secure_node("normalize_raw", normalize_raw, model_secret_ref=model_key)
    secure_search = make_secure_node("search_images", search_images)
    secure_desc = make_secure_node(
        "generate_description", generate_description, model_secret_ref=perplexity_model
    )
    secure_ner = make_secure_node("ner_technologies", ner_technologies, model_secret_ref=model_key)
    secure_verify = make_secure_node("verify_technologies_bidi", verify_technologies_bidi)
    secure_create = make_secure_node(
        "create_missing_technology_articles",
        create_missing_technology_articles,
        model_secret_ref=model_key,
    )
    secure_map = make_secure_node("map_attributes", map_attributes, model_secret_ref=model_key)

    graph.add_node("init_credentials", secure_init)
    graph.add_node("normalize_raw", secure_normalize)
    graph.add_node("search_images", secure_search)
    graph.add_node("generate_description", secure_desc)
    graph.add_node("ner_technologies", secure_ner)
    graph.add_node("verify_technologies_bidi", secure_verify)
    graph.add_node("create_missing_technology_articles", secure_create)
    graph.add_node("map_attributes", secure_map)

    graph.add_edge(START, "init_credentials")
    graph.add_edge("init_credentials", "normalize_raw")
    graph.add_edge("normalize_raw", "search_images")
    graph.add_edge("search_images", "generate_description")
    graph.add_edge("generate_description", "ner_technologies")
    graph.add_edge("ner_technologies", "verify_technologies_bidi")

    graph.add_conditional_edges(
        "verify_technologies_bidi",
        should_create_technologies,
        {"create": "create_missing_technology_articles", "skip": "map_attributes"},
    )

    graph.add_edge("create_missing_technology_articles", "map_attributes")
    graph.add_edge("map_attributes", END)

    return graph.compile()


__all__ = ["build_enricher_graph"]
