"""Search suggestions node (agent wrapper).

Implementation lives in `contextrouter.cortex.steps.rag_retrieval.suggest` to keep direct-mode
free of registration side effects and avoid logic duplication.
"""

from __future__ import annotations

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState

from ...steps.rag_retrieval.suggest import (
    generate_search_suggestions as _generate_search_suggestions,
)


class SuggestAgent(BaseAgent):
    """Class wrapper for suggest node (strict: nodes are classes)."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await _generate_search_suggestions(state)


__all__ = ["SuggestAgent"]
