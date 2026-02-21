"""
news_engine graph - main entry point.

Routes to subgraphs based on intent:
- harvest → Fetch news via Perplexity/Serper
- archivist → Filter and validate news
- showrunner → Create editorial plan
- agents → Generate posts
- full_pipeline → All steps in sequence
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from contextrouter.core.registry import register_graph

from .agents import create_agents_subgraph
from .archivist import create_archivist_subgraph
from .harvest import create_harvest_subgraph
from .showrunner import create_showrunner_subgraph
from .state import NewsEngineState

logger = logging.getLogger(__name__)


async def route_intent_node(state: NewsEngineState) -> Dict[str, Any]:
    """Route to appropriate subgraph based on intent."""
    intent = state.get("intent", "harvest")
    logger.info("news_engine routing intent: %s", intent)
    return {"intent": intent}


def _get_next_node(state: NewsEngineState) -> str:
    """Determine next node based on intent."""
    intent = state.get("intent", "harvest")
    routing = {
        "harvest": "harvest",
        "archivist": "archivist",
        "showrunner": "showrunner",
        "agents": "agents",
        "full_pipeline": "harvest",  # Start from beginning
    }
    return routing.get(intent, END)


def _after_harvest(state: NewsEngineState) -> str:
    """After harvest, continue to archivist if full_pipeline."""
    intent = state.get("intent", "")
    if intent == "full_pipeline":
        return "archivist"
    return END


def _after_archivist(state: NewsEngineState) -> str:
    """After archivist, continue to showrunner if full_pipeline."""
    intent = state.get("intent", "")
    if intent == "full_pipeline":
        return "showrunner"
    return END


def _after_showrunner(state: NewsEngineState) -> str:
    """After showrunner, continue to agents if full_pipeline."""
    intent = state.get("intent", "")
    if intent == "full_pipeline":
        return "agents"
    return END


@register_graph("news_engine")
def build_news_engine_graph():
    """
    Build news_engine graph with subgraphs.

    Usage:
        from contextrouter.cortex.graphs.news_engine import build_news_engine_graph

        graph = build_news_engine_graph()

        # Full pipeline
        result = await graph.ainvoke({
            "intent": "full_pipeline",
            "tenant_id": "my_news_agency",
            "prompt_overrides": {"harvester": "custom prompt..."},
        })

        # Individual steps
        result = await graph.ainvoke({
            "intent": "archivist",
            "tenant_id": "my_news_agency",
            "raw_items": [...],
        })
    """
    workflow = StateGraph(NewsEngineState)

    # Router node
    workflow.add_node("route", route_intent_node)

    # Subgraphs
    workflow.add_node("harvest", create_harvest_subgraph())
    workflow.add_node("archivist", create_archivist_subgraph())
    workflow.add_node("showrunner", create_showrunner_subgraph())
    workflow.add_node("agents", create_agents_subgraph())

    # Entry
    workflow.set_entry_point("route")

    # Initial routing
    workflow.add_conditional_edges(
        "route",
        _get_next_node,
        {
            "harvest": "harvest",
            "archivist": "archivist",
            "showrunner": "showrunner",
            "agents": "agents",
            END: END,
        },
    )

    # Pipeline progression for full_pipeline
    workflow.add_conditional_edges("harvest", _after_harvest, {"archivist": "archivist", END: END})
    workflow.add_conditional_edges(
        "archivist", _after_archivist, {"showrunner": "showrunner", END: END}
    )
    workflow.add_conditional_edges("showrunner", _after_showrunner, {"agents": "agents", END: END})
    workflow.add_edge("agents", END)

    return workflow.compile()
