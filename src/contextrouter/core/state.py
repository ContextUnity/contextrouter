"""Core (vendor-agnostic) agent state models.

During migration, `contextrouter.cortex.state` continues to define the LangGraph
TypedDict state used in production. This module introduces Pydantic-based state
models to support stronger validation and deterministic normalization in the new
framework layout.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from contextrouter.cortex.models import Citation, RetrievedDoc


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


class _BaseState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # Vendor-agnostic scratchpad for agent-specific fields.
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Optional internal auth context (Biscuit token or compatible object).
    access_token: Any | None = None


class InputState(_BaseState):
    """Input state for graph invocation (host-provided fields)."""

    messages: Sequence[BaseMessage] = Field(default_factory=list)
    session_id: str = ""
    platform: str = ""

    # Optional feature flags
    enable_suggestions: bool = True
    suggestions_model: str = "gemini-2.0-flash-lite"
    enable_web_search: bool = True
    web_allowed_domains: list[str] = Field(default_factory=list)
    max_web_results: int = 10

    no_results_prompt: str = ""
    rag_system_prompt_override: str = ""
    search_suggestions_prompt_override: str = ""

    # Output shaping
    citations_output: str = "raw"
    citations_allowed_types: list[str] = Field(default_factory=list)

    # Host-provided prompt style
    style_prompt: str = ""


class AgentState(InputState):
    """Full internal state passed between agent nodes."""

    user_query: str = ""
    user_language: str = ""
    intent: str = ""
    intent_text: str = ""
    ignore_history: bool = False

    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    search_suggestions: list[str] = Field(default_factory=list)
    should_retrieve: bool = False
    generation_complete: bool = False

    # Taxonomy and graph fields for hybrid retrieval
    taxonomy_categories: list[str] = Field(default_factory=list)
    taxonomy_concepts: list[str] = Field(default_factory=list)
    graph_facts: list[str] = Field(default_factory=list)


class OutputState(_BaseState):
    """Output state from graph invocation (UI-facing subset)."""

    messages: Sequence[BaseMessage] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    search_suggestions: list[str] = Field(default_factory=list)


__all__ = ["InputState", "AgentState", "OutputState", "get_last_user_query"]
