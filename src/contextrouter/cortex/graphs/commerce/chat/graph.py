"""
Chat subgraph definition.

Chat: PIM Chat with LLM intent detection.
Wraps Commerce graph — detects intent from user message and routes to appropriate subgraph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .nodes import (
    aggregate_response_node,
    chat_response_node,
    detect_intent_node,
    invoke_gardener_node,
    invoke_lexicon_node,
    invoke_matcher_node,
    invoke_mutator_node,
    route_to_subgraph,
    trigger_channel_sync_node,
    trigger_harvester_node,
)
from .state import ChatState

logger = logging.getLogger(__name__)


def create_chat_subgraph():
    """Create Chat subgraph for PIM Chat.

    This subgraph adds LLM intent detection on top of Commerce.
    Use this when you have natural language input from users.
    
    For programmatic access, use build_commerce_graph() directly with intent field.

    Flow:
        detect_intent → [route based on intent] → aggregate → END
    """
    graph = StateGraph(ChatState)

    # Intent detection
    graph.add_node("detect_intent", detect_intent_node)

    # Action nodes
    graph.add_node("trigger_harvester", trigger_harvester_node)
    graph.add_node("trigger_channel_sync", trigger_channel_sync_node)
    graph.add_node("invoke_matcher", invoke_matcher_node)
    graph.add_node("invoke_lexicon", invoke_lexicon_node)
    graph.add_node("invoke_gardener", invoke_gardener_node)
    graph.add_node("invoke_mutator", invoke_mutator_node)
    graph.add_node("chat_response", chat_response_node)
    graph.add_node("aggregate", aggregate_response_node)

    # Entry
    graph.add_edge(START, "detect_intent")

    # Conditional routing based on intent
    graph.add_conditional_edges(
        "detect_intent",
        route_to_subgraph,
        {
            "trigger_harvester": "trigger_harvester",
            "trigger_channel_sync": "trigger_channel_sync",
            "invoke_matcher": "invoke_matcher",
            "invoke_lexicon": "invoke_lexicon",
            "invoke_gardener": "invoke_gardener",
            "invoke_mutator": "invoke_mutator",
            "chat_response": "chat_response",
        },
    )

    # All paths lead to aggregate
    for node in [
        "trigger_harvester",
        "trigger_channel_sync",
        "invoke_matcher",
        "invoke_lexicon",
        "invoke_gardener",
        "invoke_mutator",
        "chat_response",
    ]:
        graph.add_edge(node, "aggregate")

    graph.add_edge("aggregate", END)

    return graph.compile()


async def invoke_chat(user_message: str, product_id: str = None) -> Dict[str, Any]:
    """
    Run Chat (PIM supervisor).

    Entry point for PIM Chat with LLM intent detection.
    
    Args:
        user_message: Natural language message from user
        product_id: Optional product context
        
    Returns:
        Dict with intent, response, actions, and sub_result
    """
    graph = create_chat_subgraph()

    initial_state: ChatState = {
        "user_message": user_message,
        "product_id": product_id,
        "intent": "",
        "confidence": 0.0,
        "extracted_params": {},
        "sub_task_result": None,
        "response": "",
        "actions_taken": [],
        "total_tokens": 0,
    }

    final_state = await graph.ainvoke(initial_state)

    return {
        "intent": final_state["intent"],
        "response": final_state["response"],
        "actions": final_state["actions_taken"],
        "sub_result": final_state["sub_task_result"],
    }
