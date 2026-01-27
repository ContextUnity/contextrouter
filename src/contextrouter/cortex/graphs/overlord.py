"""
Overlord Agent - LangGraph implementation (STUB).

The Supervisor: Routes intents within PIM Chat, orchestrates tasks.

Flow:
1. detect_intent - Classify user message intent
2. route_to_agent - Dispatch to appropriate agent/task
3. aggregate_response - Combine results
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# Supported intents
Intent = Literal[
    "import_products",  # Trigger Harvester
    "sync_channel",  # Push to Horoshop, etc.
    "match_products",  # Run Matcher
    "update_content",  # Run Lexicon
    "classify_taxonomy",  # Run Gardener
    "edit_product",  # Mutator (inline)
    "general_query",  # General chat
]


class OverlordState(TypedDict):
    """State for Overlord graph."""

    # Input
    user_message: str
    product_id: Optional[str]  # If in product context

    # Processing
    intent: str
    confidence: float
    extracted_params: Dict[str, Any]

    # Sub-task results
    sub_task_result: Any

    # Output
    response: str
    actions_taken: List[str]

    # Metrics
    total_tokens: int


from typing import Optional  # noqa: E402


async def detect_intent_node(state: OverlordState) -> OverlordState:
    """Classify user message intent."""
    from ..llm import get_llm

    llm = get_llm()

    # TODO: Implement intent classification with LLM
    # system_prompt = """You are a PIM assistant. Classify intent..."""
    # response = await llm.ainvoke([...])
    _ = llm  # Will be used when TODO above is implemented

    # TODO: Parse response
    state["intent"] = "general_query"
    state["confidence"] = 0.8
    state["extracted_params"] = {}

    return state


def route_to_agent(state: OverlordState) -> str:
    """Route to appropriate agent based on intent."""
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


async def trigger_harvester_node(state: OverlordState) -> OverlordState:
    """Trigger Harvester import job."""
    # TODO: Call Worker/Temporal to start import
    params = state.get("extracted_params", {})
    supplier = params.get("supplier", "all")

    logger.info(f"Would trigger Harvester for {supplier}")
    state["sub_task_result"] = {"status": "triggered", "supplier": supplier}
    state["actions_taken"] = [f"Triggered import for {supplier}"]

    return state


async def trigger_channel_sync_node(state: OverlordState) -> OverlordState:
    """Trigger channel sync (Horoshop, etc.)."""
    params = state.get("extracted_params", {})
    channel = params.get("channel", "horoshop")

    logger.info(f"Would trigger sync to {channel}")
    state["sub_task_result"] = {"status": "triggered", "channel": channel}
    state["actions_taken"] = [f"Triggered sync to {channel}"]

    return state


async def invoke_matcher_node(state: OverlordState) -> OverlordState:
    """Invoke Matcher agent."""
    from .matcher import invoke_matcher

    result = await invoke_matcher()
    state["sub_task_result"] = result
    state["actions_taken"] = [f"Matcher: {result.get('auto_linked', 0)} linked"]

    return state


async def invoke_lexicon_node(state: OverlordState) -> OverlordState:
    """Invoke Lexicon agent."""
    from .lexicon import invoke_lexicon

    result = await invoke_lexicon()
    state["sub_task_result"] = result
    state["actions_taken"] = [f"Lexicon: {result.get('drafts_created', 0)} drafts"]

    return state


async def invoke_gardener_node(state: OverlordState) -> OverlordState:
    """Invoke Gardener agent."""
    from .gardener import invoke_gardener

    result = await invoke_gardener()
    state["sub_task_result"] = result
    state["actions_taken"] = [f"Gardener: {result.get('proposals', 0)} proposals"]

    return state


async def invoke_mutator_node(state: OverlordState) -> OverlordState:
    """Invoke Mutator for product editing."""
    # Mutator runs in Commerce Django, not here
    # This just prepares the response
    state["response"] = "Opening product editor..."
    state["actions_taken"] = ["Opened Mutator"]
    return state


async def chat_response_node(state: OverlordState) -> OverlordState:
    """Generate general chat response."""

    # TODO: Use main brain graph for general queries
    state["response"] = (
        "I can help you with product management. Try:\n"
        "- 'Import products from Abris'\n"
        "- 'Match unlinked products'\n"
        "- 'Sync to Horoshop'"
    )
    return state


async def aggregate_response_node(state: OverlordState) -> OverlordState:
    """Combine all results into final response."""
    actions = state.get("actions_taken", [])
    result = state.get("sub_task_result", {})

    if actions:
        state["response"] = f"Done! {', '.join(actions)}\n\nDetails: {result}"

    return state


def create_overlord_graph() -> StateGraph:
    """Create Overlord LangGraph."""
    graph = StateGraph(OverlordState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("trigger_harvester", trigger_harvester_node)
    graph.add_node("trigger_channel_sync", trigger_channel_sync_node)
    graph.add_node("invoke_matcher", invoke_matcher_node)
    graph.add_node("invoke_lexicon", invoke_lexicon_node)
    graph.add_node("invoke_gardener", invoke_gardener_node)
    graph.add_node("invoke_mutator", invoke_mutator_node)
    graph.add_node("chat_response", chat_response_node)
    graph.add_node("aggregate", aggregate_response_node)

    # Edges
    graph.add_edge(START, "detect_intent")

    # Conditional routing
    graph.add_conditional_edges(
        "detect_intent",
        route_to_agent,
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


async def invoke_overlord(user_message: str, product_id: str = None) -> Dict[str, Any]:
    """
    Run Overlord agent (PIM supervisor).

    Entry point for PIM Chat.
    """
    graph = create_overlord_graph()

    initial_state: OverlordState = {
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
