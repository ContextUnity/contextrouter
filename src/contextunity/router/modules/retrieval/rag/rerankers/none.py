"""No-op reranker -- pass-through that preserves original retrieval order (default fallback)."""

from __future__ import annotations

from typing import override

from ..models import RetrievedDoc
from .base import BaseReranker


class NoopReranker(BaseReranker):
    """Pass-through reranker — returns documents as-is (used when reranking is disabled)."""

    @override
    async def rerank(
        self,
        *,
        query: str,
        documents: list[RetrievedDoc],
        top_n: int | None = None,
        source_type: str | None = None,
    ) -> list[RetrievedDoc]:
        """Return *documents* truncated to *top_n* without reordering."""
        _ = query, source_type
        return documents[:top_n] if top_n else documents


__all__ = ["NoopReranker"]
