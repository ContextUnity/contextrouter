"""Agent wrappers for RAG retrieval nodes.

These classes wrap functional nodes for agent-mode compatibility.
In agent-mode, LangGraph uses classes from the registry instead of functions.
"""

from __future__ import annotations

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState

from .extract import extract_user_query
from .generate import generate_response
from .intent import detect_intent
from .retrieve import retrieve_documents
from .suggest import generate_search_suggestions


class ExtractQueryAgent(BaseAgent):
    """Agent wrapper for extract_query node."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return extract_user_query(state)


class DetectIntentAgent(BaseAgent):
    """Agent wrapper for detect_intent node."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await detect_intent(state)


class RetrieveAgent(BaseAgent):
    """Agent wrapper for retrieve node."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await retrieve_documents(state)


class SuggestAgent(BaseAgent):
    """Agent wrapper for suggest node."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await generate_search_suggestions(state)


class GenerateAgent(BaseAgent):
    """Agent wrapper for generate node."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await generate_response(state)


__all__ = [
    "ExtractQueryAgent",
    "DetectIntentAgent",
    "RetrieveAgent",
    "SuggestAgent",
    "GenerateAgent",
]
