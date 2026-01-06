"""Extract-query node (agent wrapper).

Implementation lives in `contextrouter.cortex.steps.rag_retrieval.extract` to keep direct-mode
free of registration side effects and avoid logic duplication.
"""

from __future__ import annotations

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState

from ...steps.rag_retrieval.extract import extract_user_query as _extract_user_query


class ExtractQueryAgent(BaseAgent):
    """Class wrapper for extract_query node (strict: nodes are classes)."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return _extract_user_query(state)


__all__ = ["ExtractQueryAgent"]
