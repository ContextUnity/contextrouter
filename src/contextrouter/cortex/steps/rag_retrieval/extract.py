"""Extract-query step (pure function)."""

from __future__ import annotations

import logging
import time

from contextrouter.cortex import AgentState, get_last_user_query

from ...nodes.utils import pipeline_log

logger = logging.getLogger(__name__)


def extract_user_query(state: AgentState) -> dict[str, object]:
    """Extract the latest user query and initialize default state fields."""
    t0 = time.perf_counter()
    messages = state.get("messages", [])
    user_query = (get_last_user_query(messages) or "").strip()

    logger.debug("Extract: messages=%d query=%s", len(messages), (user_query or "")[:80])

    out: dict[str, object] = {
        "user_query": user_query,
        "user_language": "",
        "should_retrieve": bool(user_query),
        "search_suggestions": [],
        "intent": "rag_and_web",
        "intent_text": user_query,
        "ignore_history": False,
        "retrieval_queries": [user_query] if user_query else [],
        "retrieved_docs": [],
        "citations": [],
        "generation_complete": False,
    }

    pipeline_log(
        "extract_query",
        user_query=out.get("user_query"),
        should_retrieve=out.get("should_retrieve"),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )

    return out


__all__ = ["extract_user_query"]
