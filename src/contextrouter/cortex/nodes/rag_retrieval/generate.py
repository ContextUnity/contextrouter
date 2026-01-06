"""Final generation node (agent wrapper).

Implementation lives in `contextrouter.cortex.steps.rag_retrieval.generate` to keep direct-mode
free of registration side effects and avoid logic duplication.
"""

from __future__ import annotations

from typing import Any

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState

from ...steps.rag_retrieval.generate import generate_response as _generate_response


class GenerateAgent(BaseAgent):
    """Class wrapper for generate node (strict: nodes are classes)."""

    async def process(self, state: AgentState) -> dict[str, Any]:
        return await _generate_response(state)


__all__ = ["GenerateAgent"]
