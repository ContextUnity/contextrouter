"""
Lexicon Agent - LangGraph implementation (STUB).

The Researcher: Fills wiki/content for new entities.

Flow:
1. detect_missing - Find brands/technologies without descriptions
2. research - Web search via Perplexity
3. draft_content - LLM generates content
4. create_draft - Create Wagtail draft page
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


class LexiconState(TypedDict):
    """State for Lexicon graph."""

    # Input
    entity_types: List[str]  # ["brand", "technology", "material"]
    batch_size: int

    # Processing
    missing_entities: List[Dict[str, Any]]
    research_results: List[Dict[str, Any]]
    drafted_pages: List[Dict[str, Any]]

    # Output
    drafts_created: int

    # Metrics
    total_tokens: int
    errors: List[str]


async def detect_missing_node(state: LexiconState) -> LexiconState:
    """Find entities without descriptions."""
    # TODO: Query Commerce DB for brands/technologies with empty descriptions
    # SELECT * FROM catalogue_brand WHERE description IS NULL LIMIT batch_size

    state["missing_entities"] = []
    logger.info(f"Found {len(state['missing_entities'])} entities needing content")
    return state


async def research_web_node(state: LexiconState) -> LexiconState:
    """Research entities via Perplexity API."""
    # TODO: Implement Perplexity search
    # For each entity:
    #   - Search "{brand_name} outdoor equipment company"
    #   - Extract key facts, founding year, products, etc.

    state["research_results"] = []
    return state


async def draft_content_node(state: LexiconState) -> LexiconState:
    """Generate content draft using LLM."""
    from ..llm import get_llm

    if not state["research_results"]:
        return state

    llm = get_llm()  # noqa: F841 - TODO: implement

    drafted = []
    for research in state["research_results"]:
        # TODO: LLM generates structured content
        # prompt = f"""
        # Based on the following research, create a product wiki page:
        # Entity: {research.get("name")}
        # Type: {research.get("type")}
        # Generate structured JSON with sections.
        # """
        # response = await llm.ainvoke(prompt)
        _ = llm  # Use llm to silence linter (will be used in TODO above)

        drafted.append(
            {
                "entity_id": research.get("id"),
                "entity_name": research.get("name"),
                "title": f"{research.get('name')} - Product Wiki",
                "body": "TODO: Generated content",
            }
        )

    state["drafted_pages"] = drafted
    return state


async def create_wagtail_draft_node(state: LexiconState) -> LexiconState:
    """Create draft pages in Wagtail CMS."""
    # TODO: Call Wagtail API or direct DB insert
    # Creates pages in "pending review" state

    for draft in state.get("drafted_pages", []):
        logger.info(f"Would create draft: {draft['title']}")

    state["drafts_created"] = len(state.get("drafted_pages", []))
    return state


def create_lexicon_graph() -> StateGraph:
    """Create Lexicon LangGraph."""
    graph = StateGraph(LexiconState)

    graph.add_node("detect_missing", detect_missing_node)
    graph.add_node("research", research_web_node)
    graph.add_node("draft_content", draft_content_node)
    graph.add_node("create_draft", create_wagtail_draft_node)

    graph.add_edge(START, "detect_missing")
    graph.add_edge("detect_missing", "research")
    graph.add_edge("research", "draft_content")
    graph.add_edge("draft_content", "create_draft")
    graph.add_edge("create_draft", END)

    return graph.compile()


async def invoke_lexicon(entity_types: List[str] = None, batch_size: int = 10) -> Dict[str, Any]:
    """Run Lexicon agent."""
    if entity_types is None:
        entity_types = ["brand", "technology", "material"]

    graph = create_lexicon_graph()

    initial_state: LexiconState = {
        "entity_types": entity_types,
        "batch_size": batch_size,
        "missing_entities": [],
        "research_results": [],
        "drafted_pages": [],
        "drafts_created": 0,
        "total_tokens": 0,
        "errors": [],
    }

    final_state = await graph.ainvoke(initial_state)

    return {
        "entities_found": len(final_state["missing_entities"]),
        "drafts_created": final_state["drafts_created"],
    }
