"""Retrieve step (pure function)."""

from __future__ import annotations

import logging

from contextrouter.cortex import AgentState

logger = logging.getLogger(__name__)


async def retrieve_documents(state: AgentState) -> dict[str, object]:
    """Retrieve documents for the current query.

    RAG is an optional capability. If disabled, retrieval returns empty context.
    """

    if state.get("enable_rag") is False:
        return {"retrieved_docs": [], "citations": [], "should_retrieve": False, "graph_facts": []}

    # Lazy import: avoids pulling RAG modules when capability is disabled.
    from contextrouter.modules.retrieval.rag import RagPipeline

    queries = state.get("retrieval_queries", [])
    logger.debug("Retrieve: queries=%d", len(queries))

    try:
        res = await RagPipeline().execute(state)
        logger.debug(
            "Retrieve complete: docs=%d citations=%d facts=%d",
            len(res.retrieved_docs),
            len(res.citations),
            len(res.graph_facts),
        )
        return {
            "retrieved_docs": res.retrieved_docs,
            "citations": res.citations,
            "graph_facts": res.graph_facts,
            "should_retrieve": False,
        }
    except Exception:
        logger.exception("Retrieval failed")
        return {"retrieved_docs": [], "citations": [], "should_retrieve": False, "graph_facts": []}


__all__ = ["retrieve_documents"]
