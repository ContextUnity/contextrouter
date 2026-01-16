"""Postgres knowledge store models and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field

from contextrouter.core.types import StructData


class TaxonomyPath(BaseModel):
    """A hierarchical taxonomy scope encoded as an ltree-compatible string."""

    path: str = Field(..., description="ltree path, e.g. book.chapter_03.faith")


class GraphNode(BaseModel):
    id: str
    content: str
    embedding: List[float] | None = None
    node_kind: str = "concept"  # 'chunk' | 'concept'
    source_type: str | None = None  # video/book/qa/web/knowledge for chunks
    source_id: str | None = None
    taxonomy_path: str | None = None  # ltree string
    metadata: StructData = Field(default_factory=dict)
    title: str | None = None
    keywords_text: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str  # e.g., supports/contradicts/requires_action/relates_to
    weight: float = 1.0
    metadata: StructData = Field(default_factory=dict)
    tenant_id: str | None = None


class SearchResult(BaseModel):
    node: GraphNode
    score: float
    vector_score: float | None = None
    text_score: float | None = None
    connected_nodes: List[GraphNode] = Field(default_factory=list)


class KnowledgeStoreInterface(ABC):
    @abstractmethod
    async def upsert_graph(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        *,
        tenant_id: str,
        user_id: str | None = None,
    ) -> None:
        """Store nodes+edges transactionally."""

    @abstractmethod
    async def hybrid_search(
        self,
        *,
        query_text: str,
        query_vec: List[float],
        candidate_k: int = 50,
        limit: int = 8,
        scope: TaxonomyPath | None = None,
        source_types: List[str] | None = None,
        graph_depth: int = 2,
        allowed_relations: List[str] | None = None,
        fusion: str = "weighted",
        rrf_k: int = 60,
        vector_weight: float = 0.8,
        text_weight: float = 0.2,
        tenant_id: str,
        user_id: str | None = None,
    ) -> List[SearchResult]:
        """Hybrid search + optional graph enrichment."""


__all__ = [
    "TaxonomyPath",
    "GraphNode",
    "GraphEdge",
    "SearchResult",
    "KnowledgeStoreInterface",
]
