"""
Chat node implementations.

LLM intent detection and routing for PIM Chat.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .state import ChatState

logger = logging.getLogger(__name__)


async def detect_intent_node(state: ChatState) -> Dict[str, Any]:
    """Classify user message intent using LLM."""
    from ....llm import get_llm

    llm = get_llm()

    # TODO: Implement intent classification with LLM
    # system_prompt = """You are a PIM assistant. Classify intent..."""
    # response = await llm.ainvoke([...])
    _ = llm  # Will be used when TODO above is implemented

    # TODO: Parse response
    return {
        "intent": "general_query",
        "confidence": 0.8,
        "extracted_params": {},
    }


def route_to_subgraph(state: ChatState) -> str:
    """Route to appropriate subgraph based on intent."""
    intent = state.get("intent", "general_query")

    routes = {
        "import_products": "trigger_harvester",
        "sync_channel": "trigger_channel_sync",
        "match_products": "invoke_matcher",
        "update_content": "invoke_lexicon",
        "classify_taxonomy": "invoke_gardener",
        "edit_product": "invoke_mutator",
        "general_query": "chat_response",
    }

    return routes.get(intent, "chat_response")


async def trigger_harvester_node(state: ChatState) -> Dict[str, Any]:
    """Trigger Harvester import job."""
    params = state.get("extracted_params", {})
    supplier = params.get("supplier", "all")

    logger.info(f"Would trigger Harvester for {supplier}")
    return {
        "sub_task_result": {"status": "triggered", "supplier": supplier},
        "actions_taken": [f"Triggered import for {supplier}"],
    }


async def trigger_channel_sync_node(state: ChatState) -> Dict[str, Any]:
    """Trigger channel sync (Horoshop, etc.)."""
    params = state.get("extracted_params", {})
    channel = params.get("channel", "horoshop")

    logger.info(f"Would trigger sync to {channel}")
    return {
        "sub_task_result": {"status": "triggered", "channel": channel},
        "actions_taken": [f"Triggered sync to {channel}"],
    }


async def invoke_matcher_node(state: ChatState) -> Dict[str, Any]:
    """Invoke Matcher via Commerce graph."""
    from ..graph import build_commerce_graph

    graph = build_commerce_graph()
    result = await graph.ainvoke({"intent": "match_products"})
    return {
        "sub_task_result": result,
        "actions_taken": ["Matcher: invoked via Commerce graph"],
    }


async def invoke_lexicon_node(state: ChatState) -> Dict[str, Any]:
    """Invoke Lexicon via Commerce graph."""
    from ..graph import build_commerce_graph

    graph = build_commerce_graph()
    result = await graph.ainvoke({"intent": "generate_content"})
    return {
        "sub_task_result": result,
        "actions_taken": ["Lexicon: invoked via Commerce graph"],
    }


async def invoke_gardener_node(state: ChatState) -> Dict[str, Any]:
    """Invoke Gardener via Commerce graph."""
    from ..graph import build_commerce_graph

    graph = build_commerce_graph()
    result = await graph.ainvoke({"intent": "enrich"})
    return {
        "sub_task_result": result,
        "actions_taken": ["Gardener: invoked via Commerce graph"],
    }


async def invoke_mutator_node(state: ChatState) -> Dict[str, Any]:
    """Invoke Mutator for product editing."""
    return {
        "response": "Opening product editor...",
        "actions_taken": ["Opened Mutator"],
    }


async def chat_response_node(state: ChatState) -> Dict[str, Any]:
    """Generate general chat response."""
    return {
        "response": (
            "I can help you with product management. Try:\n"
            "- 'Import products from Abris'\n"
            "- 'Match unlinked products'\n"
            "- 'Sync to Horoshop'"
        ),
    }


async def aggregate_response_node(state: ChatState) -> Dict[str, Any]:
    """Combine all results into final response."""
    actions = state.get("actions_taken", [])
    result = state.get("sub_task_result", {})

    if actions:
        return {"response": f"Done! {', '.join(actions)}\n\nDetails: {result}"}
    return {}
