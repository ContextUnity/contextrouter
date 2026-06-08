"""Extract-query step — LLM-driven search query extraction from user messages (pure graph node)."""

from __future__ import annotations

import time
from typing import ClassVar

from contextunity.core import get_contextunit_logger
from pydantic import BaseModel, ConfigDict

from contextunity.router.cortex.core_state import get_last_user_query
from contextunity.router.cortex.types import GraphState, StateUpdate
from contextunity.router.cortex.utils.pipeline import pipeline_log


class ExtractQueryConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


logger = get_contextunit_logger(__name__)


def extract_user_query(state: GraphState) -> StateUpdate:
    """Extract the latest user query and initialize pipeline state defaults.

    Reads the conversation ``messages`` list, extracts the last human
    message via ``get_last_user_query``, and populates the ``dynamic``
    state dict with zero-value defaults for all downstream pipeline
    fields (``intent``, ``retrieval_queries``, ``citations``, etc.).

    Args:
        state: Current graph execution state with ``messages``.

    Returns:
        State update with initialized ``dynamic`` dict.
    """
    t0 = time.perf_counter()
    messages = state.get("messages", [])
    user_query = (get_last_user_query(messages) or "").strip()

    logger.debug("Extract: messages=%d query=%s", len(messages), (user_query or "")[:80])

    out: dict[str, object] = {
        "user_query": user_query,
        "user_language": "",
        "should_retrieve": bool(user_query),
        "search_suggestions": [],
        "intent": "rag",
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

    return {"dynamic": out}


__all__ = ["extract_user_query"]
