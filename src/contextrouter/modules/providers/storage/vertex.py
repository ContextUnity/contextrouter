"""Vertex provider (storage).

Infrastructure boundary for Vertex AI:
- IRead: retrieval/search (Vertex AI Search)
- IWrite: ingestion sink (future)

Uses ContextUnit protocol for data transport.
"""

from __future__ import annotations

import logging
from typing import Any

from contextcore import ContextToken, ContextUnit

from contextrouter.core.interfaces import BaseProvider, IRead, IWrite, secured
from contextrouter.modules.retrieval.rag.models import RetrievedDoc

logger = logging.getLogger(__name__)


# Shared utilities are now in vertex_search.py to avoid circular imports
# Import them here for backward compatibility
def endpoint_for_location(location: str) -> str:
    """Resolve Discovery Engine API endpoint (re-exported from vertex_search)."""
    from .vertex_search import _endpoint_for_location

    return _endpoint_for_location(location)


def parse_search_result(result: object) -> RetrievedDoc | None:
    """Parse Discovery Engine search result (re-exported from vertex_search)."""
    from .vertex_search import _parse_search_result

    return _parse_search_result(result)


class VertexProvider(BaseProvider, IRead, IWrite):
    @secured()
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: ContextToken,
    ) -> list[ContextUnit]:
        # Lazy import to avoid circular dependency
        from .vertex_search import search_vertex_ai_async

        source_type = None
        if isinstance(filters, dict):
            v = filters.get("source_type")
            source_type = v if isinstance(v, str) and v.strip() else None

        docs = await search_vertex_ai_async(
            query=query, max_results=int(limit), source_type_filter=source_type
        )
        out: list[ContextUnit] = []
        for d in docs:
            unit = ContextUnit(
                payload={"content": d, "source": "vertex"},
                provenance=["provider:vertex"],
            )
            out.append(unit)
        return out

    @secured()
    async def write(self, data: ContextUnit, *, token: ContextToken) -> None:
        _ = data, token
        raise NotImplementedError("VertexProvider.write is not implemented yet")

    async def sink(self, unit: ContextUnit, *, token: ContextToken) -> Any:
        # Default sink behavior delegates to write().
        await self.write(unit, token=token)
        return None


__all__ = ["VertexProvider"]
