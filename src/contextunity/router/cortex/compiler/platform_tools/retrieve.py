"""Retrieve step — executes RAG search against BrainClient and formats context (pure graph node)."""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_list
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.types import GraphState, StateUpdate


class RetrieveConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True


logger = get_contextunit_logger(__name__)


async def retrieve_documents(state: GraphState) -> StateUpdate:
    """Retrieve documents for the current query."""

    dyn = state.get("dynamic", {})
    if dyn.get("enable_rag") is False:
        return {
            "dynamic": {
                "retrieved_docs": [],
                "citations": [],
                "should_retrieve": False,
                "graph_facts": [],
            }
        }

    # Lazy import: avoids pulling RAG modules when capability is disabled.
    from contextunity.router.modules.retrieval.rag import RagPipeline

    queries_raw = dyn.get("retrieval_queries", [])
    queries = [str(query) for query in queries_raw] if is_object_list(queries_raw) else []
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
            "dynamic": {
                "retrieved_docs": res.retrieved_docs,
                "citations": res.citations,
                "graph_facts": res.graph_facts,
                "should_retrieve": False,
            }
        }
    except Exception:  # graceful-degrade: tool failure returns empty result
        logger.exception("Retrieval failed")
        return {
            "dynamic": {
                "retrieved_docs": [],
                "citations": [],
                "should_retrieve": False,
                "graph_facts": [],
            }
        }


__all__ = ["retrieve_documents"]
