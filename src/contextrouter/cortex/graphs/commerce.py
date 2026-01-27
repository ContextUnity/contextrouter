"""Commerce-specific graph for ContextRouter."""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from contextrouter.core.registry import register_graph
from contextrouter.cortex.graphs.matching import MatchingNode
from contextrouter.cortex.state import AgentState, InputState, OutputState


class CommerceOrchestrator:
    """
    Handles commerce-specific intents:
    - harvest: Trigger data collection
    - search_products: Find products in the catalog
    - match_item: Link external item to internal catalog
    """

    async def route_intent(self, state: AgentState) -> Dict[str, Any]:
        intent = state.get("intent", "search")
        if intent == "harvest":
            return {"next": "harvest_data"}
        elif intent == "match":
            return {"next": "match_item"}
        return {"next": "search_catalog"}

    async def harvest_node(self, state: AgentState) -> Dict[str, Any]:
        # In a real scenario, this would send a task to ContextWorker/Temporal
        # For now, we simulate a trigger
        supplier = state.get("supplier", "default")
        return {"response": f"Triggered harvest for {supplier}. Check status in AG-UI."}


@register_graph("commerce")
def build_commerce_graph():
    workflow = StateGraph(AgentState, input=InputState, output=OutputState)

    orchestrator = CommerceOrchestrator()
    matcher = MatchingNode()

    workflow.add_node("route", orchestrator.route_intent)
    workflow.add_node("harvest_data", orchestrator.harvest_node)
    workflow.add_node("match_item", matcher.process)

    workflow.add_edge(START, "route")

    workflow.add_conditional_edges(
        "route",
        lambda x: x["next"],
        {
            "harvest_data": "harvest_data",
            "match_item": "match_item",
            "search_catalog": END,  # Placeholder for retrieval
        },
    )

    workflow.add_edge("harvest_data", END)
    workflow.add_edge("match_item", END)

    return workflow
