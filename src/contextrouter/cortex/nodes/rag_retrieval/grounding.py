"""Grounding node (class wrapper).

Implementation lives in `contextrouter.cortex.steps.rag_retrieval.grounding` to keep direct-mode
graph free of registry registration side effects.
"""

from __future__ import annotations

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState
from contextrouter.cortex.steps.rag_retrieval.grounding import generate_with_native_grounding


class GroundingAgent(BaseAgent):
    """Class wrapper for grounding step (strict: nodes are classes)."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await generate_with_native_grounding(state)


__all__ = ["GroundingAgent"]
