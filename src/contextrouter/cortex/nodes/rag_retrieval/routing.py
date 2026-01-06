"""Routing node.

This node decides whether the graph should run retrieval or skip directly to
generation.
"""

from __future__ import annotations

import logging
from typing import Literal

from contextrouter.core import BaseAgent
from contextrouter.cortex import AgentState
from contextrouter.cortex.nodes.utils import pipeline_log

logger = logging.getLogger(__name__)


class RoutingAgent(BaseAgent):
    """Class wrapper for routing decision (strict: nodes are classes)."""

    def route(self, state: AgentState) -> Literal["retrieve", "skip_retrieve"]:
        intent = state.get("intent", "rag_and_web")

        if intent != "rag_and_web":
            logger.debug("Route: intent=%s -> skip_retrieve", intent)
            pipeline_log(
                "route",
                intent=intent,
                decision="skip_retrieve",
                should_retrieve=state.get("should_retrieve"),
                has_docs=bool(state.get("retrieved_docs")),
            )
            return "skip_retrieve"

        if state.get("should_retrieve", False) and not state.get("retrieved_docs"):
            logger.debug("Route: intent=rag_and_web should_retrieve=True -> retrieve")
            pipeline_log(
                "route",
                intent="rag_and_web",
                decision="retrieve",
                should_retrieve=state.get("should_retrieve"),
                has_docs=False,
            )
            return "retrieve"

        logger.debug(
            "Route: intent=%s -> skip_retrieve (has_docs=%s)",
            intent,
            bool(state.get("retrieved_docs")),
        )
        pipeline_log(
            "route",
            intent=intent,
            decision="skip_retrieve",
            should_retrieve=state.get("should_retrieve"),
            has_docs=bool(state.get("retrieved_docs")),
        )
        return "skip_retrieve"

    async def process(self, state: AgentState) -> dict[str, object]:
        # Not used by direct-mode graph wiring; reserved for dynamic graphs.
        return {"route": self.route(state)}


__all__ = ["RoutingAgent"]
