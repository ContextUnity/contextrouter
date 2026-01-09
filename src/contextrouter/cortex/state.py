"""LangGraph state definitions for the RAG agent.

This module defines the state schema used by the LangGraph graph.
Following LangGraph conventions, we use TypedDict for state definition.

Classes:
    InputState: Input state for graph invocation (host-provided fields).
    AgentState: Full state passed between graph nodes.
    OutputState: Output state returned from graph invocation.

The state flows through the graph as:
    InputState -> AgentState (enriched by nodes) -> OutputState

Key Fields:
    - messages: Conversation history (managed by LangGraph add_messages reducer).
    - session_id, platform: Identifiers for tracing and persistence.
    - user_query, intent, retrieval_queries: Set by detect_intent node.
    - retrieved_docs, citations: Set by retrieve node.
    - search_suggestions: Set by suggest node.
"""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from .models import Citation, RetrievedDoc


def get_last_user_query(messages: Sequence[BaseMessage]) -> str:
    """Extract the text of the last HumanMessage from a list of messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
        # Handle dict format if needed (sometimes used in internal state)
        if isinstance(msg, dict) and msg.get("type") == "human":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


class InputState(TypedDict, total=False):
    """Input state for graph invocation."""

    messages: Sequence[BaseMessage]
    session_id: str
    platform: str

    # Optional feature flags
    enable_suggestions: bool
    suggestions_model: str
    enable_web_search: bool
    enable_rag: bool
    web_allowed_domains: list[str]
    max_web_results: int

    no_results_prompt: str
    rag_system_prompt_override: str
    search_suggestions_prompt_override: str

    # Grounding filter (hard filter for VertexAISearch, e.g., "source_type: ANY('book', 'video')")
    rag_filter: str

    # Output shaping
    citations_output: str
    citations_allowed_types: list[str]

    # Host-provided prompt style
    style_prompt: str


class AgentState(InputState, total=False):
    """State for the RAG agent graph.

    Attributes:
        messages: Conversation messages (managed by LangGraph add_messages reducer)
        session_id: Session identifier for persistence
        platform: Platform source (telegram, api, website)
        user_query: Current user query being processed
        retrieved_docs: Documents retrieved from Vertex AI Search
        citations: Formatted citations for the response
        should_retrieve: Whether retrieval is needed
        generation_complete: Whether answer generation is complete
        taxonomy_categories: Taxonomy categories mapped from user query
        taxonomy_concepts: Canonical concepts from taxonomy for graph lookup
        graph_facts: Knowledge graph facts (non-citation facts)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    platform: str
    user_query: str
    user_language: str
    intent: str
    intent_text: str
    ignore_history: bool
    retrieval_queries: list[str]
    retrieved_docs: list[RetrievedDoc]
    citations: list[Citation]
    search_suggestions: list[str]
    enable_suggestions: bool
    suggestions_model: str
    enable_web_search: bool
    web_allowed_domains: list[str]
    max_web_results: int
    should_retrieve: bool
    generation_complete: bool

    # Taxonomy and graph fields for hybrid retrieval
    taxonomy_categories: list[str]
    taxonomy_concepts: list[str]
    graph_facts: list[str]

    # Grounding fields (for vertex_grounding provider)
    grounding_response: str
    grounding_citations: list[Citation]

    no_results_prompt: str
    style_prompt: str
    rag_system_prompt_override: str
    search_suggestions_prompt_override: str


class OutputState(TypedDict):
    """Output state from graph invocation."""

    messages: Sequence[BaseMessage]
    citations: list[Citation]
    search_suggestions: list[str]
