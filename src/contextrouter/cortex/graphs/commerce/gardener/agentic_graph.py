"""
Agentic Commerce Graph - ReAct pattern for product enrichment.

This graph uses LLM reasoning with tools to adaptively enrich products,
unlike the deterministic pipeline approach.

Key differences from deterministic:
- LLM decides WHICH tools to call and in WHAT order
- Can loop back if results are uncertain
- Can request human review for ambiguous cases
- Uses knowledge graph for disambiguation
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================


class AgentState(TypedDict):
    """State for agentic commerce graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    product: Dict[str, Any]  # Current product being enriched
    enrichment: Dict[str, Any]  # Accumulated enrichment data
    tenant_id: str
    trace_id: str
    iteration: int  # Safety limit for loops
    final_result: Dict[str, Any] | None


# ============================================================================
# Tools - What the agent can do
# ============================================================================


@tool
def search_taxonomy(query: str, depth: int = 3) -> Dict[str, Any]:
    """Search taxonomy tree for best matching category.

    Args:
        query: Product description or category hint
        depth: How deep in taxonomy tree to search (1-5)

    Returns:
        Dict with matched_path, confidence, alternatives
    """
    # In real implementation: calls Brain gRPC or taxonomy service
    # This is a placeholder showing the interface
    return {
        "matched_path": "Електротехніка/Кабелі та проводи/Мідні кабелі/ШВВП",
        "confidence": 0.95,
        "alternatives": [
            {"path": "Електротехніка/Кабелі та проводи/Мідні кабелі", "confidence": 0.7}
        ],
    }


@tool
def extract_entities(text: str, product_type: str | None = None) -> Dict[str, Any]:
    """Extract named entities and parameters from product text.

    Args:
        text: Product name, description, or any text
        product_type: Optional hint (e.g., "cable", "tool", "paint")

    Returns:
        Dict with extracted entities: brand, manufacturer, params, etc.
    """
    return {
        "brand": None,  # Not found
        "manufacturer_hint": "ОДЕСА",
        "params": {"cable_type": "ШВВП", "cores": 2, "section_mm2": 0.75, "length_m": 100},
    }


@tool
def query_knowledge_graph(
    entity: str, relation: str | None = None, context: str | None = None
) -> Dict[str, Any]:
    """Query Knowledge Graph for entity relationships.

    Args:
        entity: Entity name to look up
        relation: Optional specific relation type (MADE_BY, BELONGS_TO, etc.)
        context: Additional context for disambiguation

    Returns:
        Dict with found relationships and entity info
    """
    return {
        "found": True,
        "entity_type": "manufacturer",
        "canonical_name": "Одеський кабельний завод",
        "aliases": ["ОДЕСА", "ОКЗ", "Odessa Cable"],
        "relations": [
            {"type": "PRODUCES", "target": "ШВВП"},
            {"type": "LOCATED_IN", "target": "Одеса"},
        ],
    }


@tool
def update_product_enrichment(
    category_path: str | None = None,
    brand: str | None = None,
    manufacturer: str | None = None,
    params: Dict[str, Any] | None = None,
    technologies: list[str] | None = None,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Update product with enrichment data.

    Only provided fields are updated.

    Returns:
        Dict with success status and applied updates
    """
    applied = {}
    if category_path:
        applied["category_path"] = category_path
    if brand:
        applied["brand"] = brand
    if manufacturer:
        applied["manufacturer"] = manufacturer
    if params:
        applied["params"] = params
    if technologies:
        applied["technologies"] = technologies

    return {"success": True, "applied": applied, "confidence": confidence}


@tool
def request_human_review(
    reason: str,
    options: list[str] | None = None,
    priority: Literal["low", "medium", "high"] = "medium",
) -> Dict[str, Any]:
    """Request human review for ambiguous cases.

    Use when:
    - Category confidence < 0.7
    - Multiple conflicting entity matches
    - New category suggestion needed

    Args:
        reason: Why human review is needed
        options: Suggested options for human to choose from
        priority: Review priority

    Returns:
        Dict with review request status
    """
    return {
        "queued": True,
        "queue_position": 42,
        "estimated_wait_hours": 2,
        "fallback_action": "mark_as_pending",
    }


# ============================================================================
# Agent Node (LLM Reasoning)
# ============================================================================

SYSTEM_PROMPT = """You are a product enrichment specialist for an e-commerce platform.

Your task is to enrich product data with:
1. Accurate taxonomy category (specific, not generic)
2. Brand and/or manufacturer identification
3. Technical parameters extraction
4. Technology tags (materials, standards, certifications)

IMPORTANT RULES:
- Prefer SPECIFIC categories over generic ones
- Cross-check brand/manufacturer with Knowledge Graph
- If manufacturer looks like a city name, verify it's a company
- Request human review if confidence < 0.7 on critical fields
- Don't guess - use available tools to verify

Available tools:
- search_taxonomy: Find best category match
- extract_entities: Extract brand, params from text
- query_knowledge_graph: Verify entities and find relationships
- update_product_enrichment: Save enrichment results
- request_human_review: Queue for manual review if uncertain

Current product to enrich:
{product}

Previous enrichment (if any):
{enrichment}
"""


async def agent_node(state: AgentState) -> Dict[str, Any]:
    """LLM reasoning node - decides what to do next."""
    from contextrouter.cortex.llm import create_llm

    llm = create_llm()
    tools = [
        search_taxonomy,
        extract_entities,
        query_knowledge_graph,
        update_product_enrichment,
        request_human_review,
    ]
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("placeholder", "{messages}")]
    )

    chain = prompt | llm_with_tools

    response = await chain.ainvoke(
        {
            "product": state["product"],
            "enrichment": state.get("enrichment", {}),
            "messages": state["messages"],
        }
    )

    return {"messages": [response], "iteration": state["iteration"] + 1}


# ============================================================================
# Router - Continue or End
# ============================================================================


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue tool execution or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # Safety: max iterations
    if state["iteration"] >= 10:
        logger.warning("Max iterations reached, forcing end")
        return "end"

    # If LLM made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# ============================================================================
# Graph Builder
# ============================================================================


def create_agentic_gardener_graph():
    """Create agentic enrichment graph with ReAct pattern.

    Flow:
        start → agent (LLM) → [tools if needed] → agent → ... → end

    The agent loops between thinking and acting until:
    - Enrichment is complete
    - Human review is requested
    - Max iterations reached

    Usage:
        graph = create_agentic_gardener_graph()
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="Enrich this product")],
            "product": {"name": "...", "category": "..."},
            "enrichment": {},
            "tenant_id": "default",
            "trace_id": "xxx",
            "iteration": 0,
            "final_result": None
        })
    """
    # Define tools node
    tool_node = ToolNode(
        [
            search_taxonomy,
            extract_entities,
            query_knowledge_graph,
            update_product_enrichment,
            request_human_review,
        ]
    )

    # Build graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})

    # After tools, go back to agent for next reasoning step
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "create_agentic_gardener_graph",
    "AgentState",
    # Tools for external use
    "search_taxonomy",
    "extract_entities",
    "query_knowledge_graph",
    "update_product_enrichment",
    "request_human_review",
]
