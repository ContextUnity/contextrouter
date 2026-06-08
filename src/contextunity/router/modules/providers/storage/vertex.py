"""Vertex provider (storage).
Infrastructure boundary for Vertex AI:
- IRead: retrieval/search (Vertex AI Search)
- IWrite: ingestion sink (future)
Uses ContextUnit protocol for data transport.
"""

from __future__ import annotations

from typing import override

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.types import JsonDict

from contextunity.router.core.interfaces import BaseProvider, IRead, IWrite

logger = get_contextunit_logger(__name__)


class VertexProvider(BaseProvider, IRead, IWrite):
    """Vertex AI Search retrieval provider — queries Discovery Engine datastores."""

    @override
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: JsonDict | None = None,
    ) -> list[ContextUnit]:
        """Read."""
        from contextunity.router.core import get_core_config
        from contextunity.router.modules.retrieval.rag.settings import get_effective_data_store_id

        from .vertex_search import search_vertex_ai_async

        cfg = get_core_config()
        project_id = cfg.vertex.project_id
        location = (
            (getattr(cfg.vertex, "data_store_location", "") or "").strip()
            or (getattr(cfg.vertex, "discovery_engine_location", "") or "").strip()
            or "global"
        )

        if not project_id:
            logger.error("vertex.project_id must be set (TOML or env)")
            return []

        try:
            data_store_id = get_effective_data_store_id()
        except ValueError as e:
            logger.error("Failed to resolve RAG data_store_id: %s", e)
            return []

        source_type = None
        if filters:
            v = filters.get("source_type")
            source_type = v if isinstance(v, str) and v.strip() else None

        docs = await search_vertex_ai_async(
            query=query,
            max_results=int(limit),
            source_type_filter=source_type,
            project_id=project_id,
            location=location,
            data_store_id=data_store_id,
        )
        out: list[ContextUnit] = []
        for d in docs:
            unit = ContextUnit(
                payload={"content": d, "source": "vertex"},
                provenance=["provider:vertex"],
            )
            out.append(unit)
        return out

    @override
    async def write(self, data: ContextUnit) -> None:
        """Not yet implemented — reserved for future ingestion sink."""
        _ = data
        raise NotImplementedError("VertexProvider.write is not implemented yet")

    @override
    async def sink(self, unit: ContextUnit) -> None:
        """Sink."""
        # Default sink behavior delegates to write().
        await self.write(unit)
        return None


__all__ = ["VertexProvider"]
