"""
Gardener v2 graph builder.

Gardener: Normalizes products via deterministic + LLM two-pass pipeline.

Flow:
    fetch_and_prepare → normalize → write_results

Model and reasoning_effort are injected via graph config at registration time
(same pattern as sql_analytics / NSZU).
"""

from __future__ import annotations

from typing import Any

from contextcore import get_context_unit_logger
from langgraph.graph import END, START, StateGraph

from .state import GardenerState

logger = get_context_unit_logger(__name__)


def build_gardener_graph(config: dict) -> Any:
    """Build Gardener v2 graph from registration config.

    Args:
        config: Graph configuration dict with keys:
            - model_key: LLM model (e.g. 'openai/gpt-5-nano')
            - reasoning_effort: LLM reasoning effort (default: 'none')
            - fallback_keys: list of fallback model keys
    """
    from .nodes import make_fetch_and_prepare, make_normalize, make_write_results

    model_key = config.get("model_key", "")
    reasoning_effort = config.get("reasoning_effort", "none") or "none"
    config.get("fallback_keys")

    if not model_key:
        logger.error(
            "Gardener graph config missing 'model_key' — check Commerce registration. config=%s",
            config,
        )

    logger.info(
        "Building gardener graph: model=%s, reasoning=%s",
        model_key,
        reasoning_effort,
    )

    # Create nodes with config baked in (closure pattern)
    fetch_and_prepare = make_fetch_and_prepare()
    normalize = make_normalize(
        model_key=model_key,
        reasoning_effort=reasoning_effort,
    )
    write_results = make_write_results()

    graph = StateGraph(GardenerState)

    graph.add_node("fetch_and_prepare", fetch_and_prepare)
    graph.add_node("normalize", normalize)
    graph.add_node("write_results", write_results)

    graph.add_edge(START, "fetch_and_prepare")
    graph.add_edge("fetch_and_prepare", "normalize")
    graph.add_edge("normalize", "write_results")
    graph.add_edge("write_results", END)

    return graph.compile()


__all__ = ["build_gardener_graph"]
