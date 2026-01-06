"""Retrieve node (agent wrapper).

This is a generic retrieval node. It does NOT encode any specific backend (Vertex/web/etc).
Backend selection and orchestration live in `contextrouter.modules.retrieval.*`.
"""

from __future__ import annotations

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState

from ...steps.rag_retrieval.retrieve import (
    retrieve_documents as _retrieve_documents,
)


class RetrieveAgent(BaseAgent):
    """Class wrapper for retrieve node (strict: nodes are classes)."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await _retrieve_documents(state)


__all__ = ["RetrieveAgent"]
