"""Intent detection node (agent wrapper).

Implementation lives in `contextrouter.cortex.steps.rag_retrieval.intent` to keep direct-mode
free of registration side effects and avoid logic duplication.
"""

from __future__ import annotations

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState

from ...steps.rag_retrieval.intent import detect_intent as _detect_intent


class DetectIntentAgent(BaseAgent):
    """Class wrapper for detect_intent node (strict: nodes are classes)."""

    async def process(self, state: AgentState) -> dict[str, object]:
        return await _detect_intent(state)


__all__ = ["DetectIntentAgent"]
