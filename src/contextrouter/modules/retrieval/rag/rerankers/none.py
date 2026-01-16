"""No-op reranker."""

from __future__ import annotations

from ..models import RetrievedDoc
from .base import BaseReranker


class NoopReranker(BaseReranker):
    async def rerank(
        self,
        *,
        query: str,
        documents: list[RetrievedDoc],
        top_n: int | None = None,
        source_type: str | None = None,
    ) -> list[RetrievedDoc]:
        _ = query, source_type
        return documents[:top_n] if top_n else documents


__all__ = ["NoopReranker"]
