"""Reranker factory."""

from __future__ import annotations

from ..settings import RagRetrievalSettings
from .base import BaseReranker
from .none import NoopReranker
from .vertex import VertexReranker


def get_reranker(*, cfg: RagRetrievalSettings, provider: str) -> BaseReranker:
    if not cfg.reranking_enabled:
        return NoopReranker()

    rerank_provider = cfg.rerank_provider
    if rerank_provider is None:
        if provider == "postgres":
            return NoopReranker()
        rerank_provider = "vertex"

    if rerank_provider == "vertex":
        return VertexReranker()

    # Unknown reranker: fail closed (no rerank)
    return NoopReranker()


__all__ = ["get_reranker", "BaseReranker"]
