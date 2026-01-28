"""
State definitions for news_engine graph.
"""

from typing import Any, TypedDict


class PromptOverrides(TypedDict, total=False):
    """Prompt overrides passed from client."""

    harvester: str
    archivist: str
    showrunner: str
    base_prompt: str  # Base prompt for all agents
    agents: dict[str, str]  # agent_name -> prompt


class NewsEngineState(TypedDict, total=False):
    """State flowing through the news_engine graph."""

    # Input
    tenant_id: str
    intent: str  # "harvest" | "archivist" | "showrunner" | "agents" | "full_pipeline"
    prompt_overrides: PromptOverrides

    # Harvest input/output
    search_queries: list[str]
    raw_items: list[dict[str, Any]]
    harvest_source: str  # "perplexity" | "serper"
    harvest_errors: list[str]

    # Archivist input/output
    facts: list[dict[str, Any]]
    rejected_count: int
    duplicate_count: int

    # Showrunner input/output
    editorial_plan: dict[str, Any]
    selected_stories: list[dict[str, Any]]

    # Agents input/output
    stories: list[dict[str, Any]]
    posts: list[dict[str, Any]]
    generation_errors: list[str]

    # RAG context
    similar_facts: list[dict[str, Any]]
    similar_posts: list[dict[str, Any]]

    # Metadata
    trace_id: str
    result: dict[str, Any]
