"""Postgres knowledge store implementation (pgvector + ltree)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import override

import psycopg
from contextunity.core.exceptions import SecurityError
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from contextunity.router.core.types import StructData, coerce_struct_data

from .models import GraphEdge, GraphNode, KnowledgeStoreInterface, SearchResult, TaxonomyPath

type DictRow = dict[str, object]
type DictConnection = psycopg.AsyncConnection[DictRow]
type _DictPool = AsyncConnectionPool[DictConnection]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _required_str(value: object, *, default: str = "") -> str:
    if isinstance(value, str):
        return value
    return default if value is None else str(value)


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_struct_data(value: object) -> StructData:
    if value is None:
        return {}
    coerced = coerce_struct_data(value)
    if isinstance(coerced, dict):
        return coerced
    return {}


def _format_vector(vec: list[float]) -> str:
    """Serialize a float list to pgvector literal ``[0.12,0.34,...]``."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


class PostgresKnowledgeStore(KnowledgeStoreInterface):
    """pgvector + ltree knowledge store backed by an async connection pool."""

    def __init__(
        self,
        *,
        dsn: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        """Create a deferred ``AsyncConnectionPool`` with the given *dsn* and size bounds."""
        self._pool: _DictPool = AsyncConnectionPool(
            dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
            open=False,
        )

    async def _get_pool(self) -> _DictPool:
        """Return the connection pool, lazily opening it on first call."""
        if getattr(self._pool, "closed", True):
            await self._pool.open()
        return self._pool

    @override
    async def upsert_graph(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        *,
        tenant_id: str,
        user_id: str | None = None,
    ) -> None:
        """Insert or update langgraph graph definitions."""
        if not tenant_id:
            raise SecurityError("tenant_id is required")
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                setattr(conn, "row_factory", dict_row)
                for node in nodes:
                    _ = await conn.execute(
                        """
                        INSERT INTO knowledge_nodes (
                            id, tenant_id, user_id, node_kind, source_type, source_id, title,
                            content, struct_data, keywords_text, content_hash, taxonomy_path, embedding
                        )
                        VALUES (
                            %(id)s, %(tenant_id)s, %(user_id)s, %(node_kind)s, %(source_type)s, %(source_id)s,
                            %(title)s, %(content)s, %(struct_data)s, %(keywords_text)s, %(content_hash)s, %(taxonomy_path)s,
                            %(embedding)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            struct_data = EXCLUDED.struct_data,
                            keywords_text = EXCLUDED.keywords_text,
                            taxonomy_path = EXCLUDED.taxonomy_path,
                            embedding = EXCLUDED.embedding
                        """,
                        {
                            "id": node.id,
                            "tenant_id": tenant_id,
                            "user_id": user_id,
                            "node_kind": node.node_kind,
                            "source_type": node.source_type,
                            "source_id": node.source_id,
                            "title": node.title,
                            "content": node.content,
                            "struct_data": node.metadata,
                            "keywords_text": node.keywords_text,
                            "content_hash": None,
                            "taxonomy_path": node.taxonomy_path,
                            "embedding": _format_vector(node.embedding) if node.embedding else None,
                        },
                    )
                for edge in edges:
                    _ = await conn.execute(
                        """
                        INSERT INTO knowledge_edges (
                            tenant_id, source_id, target_id, relation, weight, metadata
                        )
                        VALUES (%(tenant_id)s, %(source_id)s, %(target_id)s, %(relation)s, %(weight)s, %(metadata)s)
                        ON CONFLICT (tenant_id, source_id, target_id, relation) DO UPDATE SET
                            weight = EXCLUDED.weight,
                            metadata = EXCLUDED.metadata
                        """,
                        {
                            "tenant_id": tenant_id,
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "relation": edge.relation,
                            "weight": edge.weight,
                            "metadata": edge.metadata,
                        },
                    )

    @override
    async def hybrid_search(
        self,
        *,
        query_text: str,
        query_vec: list[float],
        candidate_k: int = 50,
        limit: int = 8,
        scope: TaxonomyPath | None = None,
        source_types: list[str] | None = None,
        graph_depth: int = 2,
        allowed_relations: list[str] | None = None,
        fusion: str = "weighted",
        rrf_k: int = 60,
        vector_weight: float = 0.8,
        text_weight: float = 0.2,
        tenant_id: str,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        """Run vector + full-text search, fuse scores, and return ranked nodes.

        Retrieves ``candidate_k`` hits from each channel (cosine distance for vectors,
        ``ts_rank_cd`` for text), merges them via the chosen ``fusion`` strategy
        (weighted linear or reciprocal rank fusion), then hydrates the top ``limit``
        node objects.

        Raises:
            ValueError: If ``tenant_id`` is empty.
        """
        _ = graph_depth, allowed_relations
        if not tenant_id:
            raise SecurityError("tenant_id is required")
        if candidate_k <= 0 or limit <= 0:
            return []
        candidate_k = max(1, candidate_k)
        limit = min(limit, candidate_k)

        pool = await self._get_pool()
        async with pool.connection() as conn:
            setattr(conn, "row_factory", dict_row)
            vector_hits = await self._fetch_vector_hits(
                conn=conn,
                tenant_id=tenant_id,
                user_id=user_id,
                query_vec=query_vec,
                candidate_k=candidate_k,
                scope=scope,
                source_types=source_types,
            )
            text_hits = await self._fetch_text_hits(
                conn=conn,
                tenant_id=tenant_id,
                user_id=user_id,
                query_text=query_text,
                candidate_k=candidate_k,
                scope=scope,
                source_types=source_types,
            )
            ranked_ids = self._fuse_results(
                vector_hits=vector_hits,
                text_hits=text_hits,
                fusion=fusion,
                rrf_k=rrf_k,
                vector_weight=vector_weight,
                text_weight=text_weight,
                limit=limit,
            )
            if not ranked_ids:
                return []
            nodes = await self._fetch_nodes(
                conn=conn, tenant_id=tenant_id, ids=[rid for rid, _ in ranked_ids]
            )
            node_map = {n.id: n for n in nodes}
            return [
                SearchResult(
                    node=node_map[rid],
                    score=score,
                    vector_score=vector_hits.get(rid),
                    text_score=text_hits.get(rid),
                )
                for rid, score in ranked_ids
                if rid in node_map
            ]

    async def _fetch_vector_hits(
        self,
        *,
        conn: DictConnection,
        tenant_id: str,
        user_id: str | None,
        query_vec: list[float],
        candidate_k: int,
        scope: TaxonomyPath | None,
        source_types: list[str] | None,
    ) -> dict[str, float]:
        """Run a cosine-similarity query and return ``{node_id: score}``."""
        clauses, params = self._build_scope_filters(
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,
            source_types=source_types,
        )
        where_sql = sql.SQL(" AND ").join(clauses)
        query = (
            sql.SQL(
                """
            SELECT id, 1 - (embedding <=> %s::vector) AS vector_score
            FROM knowledge_nodes
            WHERE node_kind = 'chunk'
              AND embedding IS NOT NULL
              AND
            """
            )
            + where_sql
            + sql.SQL(
                """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            )
        )
        return await self._fetch_scores(
            conn=conn,
            sql=query,
            params=[_format_vector(query_vec), *params, _format_vector(query_vec), candidate_k],
            score_key="vector_score",
        )

    async def _fetch_text_hits(
        self,
        *,
        conn: DictConnection,
        tenant_id: str,
        user_id: str | None,
        query_text: str,
        candidate_k: int,
        scope: TaxonomyPath | None,
        source_types: list[str] | None,
    ) -> dict[str, float]:
        """Run a ``websearch_to_tsquery`` full-text search and return ``{node_id: score}``."""
        if not query_text.strip():
            return {}
        clauses, params = self._build_scope_filters(
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,
            source_types=source_types,
        )
        where_sql = sql.SQL(" AND ").join(clauses)
        query = (
            sql.SQL(
                """
            SELECT id,
                   ts_rank_cd(
                       search_vector || COALESCE(keywords_vector, ''::tsvector),
                       websearch_to_tsquery('simple', %s)
                   ) AS text_score
            FROM knowledge_nodes
            WHERE node_kind = 'chunk'
              AND (
                  search_vector || COALESCE(keywords_vector, ''::tsvector)
              ) @@ websearch_to_tsquery('simple', %s)
              AND
            """
            )
            + where_sql
            + sql.SQL(
                """
            ORDER BY text_score DESC
            LIMIT %s
            """
            )
        )
        return await self._fetch_scores(
            conn=conn,
            sql=query,
            params=[query_text, query_text, *params, candidate_k],
            score_key="text_score",
        )

    async def _fetch_scores(
        self, *, conn: DictConnection, sql: sql.Composed, params: list[object], score_key: str
    ) -> dict[str, float]:
        """Execute a scoring query and collect ``{id: score}`` from the result rows."""
        rows = await conn.execute(sql, params)
        out: dict[str, float] = {}
        async for row in rows:
            rid = _required_str(row["id"])
            score = _to_float(row.get(score_key))
            if score is None:
                continue
            out[rid] = score
        return out

    async def _fetch_nodes(
        self, *, conn: DictConnection, tenant_id: str, ids: Iterable[str]
    ) -> list[GraphNode]:
        """Hydrate ``GraphNode`` objects for the given IDs within a tenant."""
        rows = await conn.execute(
            """
            SELECT id, node_kind, source_type, source_id, title, content, struct_data, taxonomy_path, tenant_id, user_id
            FROM knowledge_nodes
            WHERE tenant_id = %s AND id = ANY(%s::text[])
            """,
            [tenant_id, list(ids)],
        )
        nodes: list[GraphNode] = []
        async for row in rows:
            nodes.append(
                GraphNode(
                    id=_required_str(row["id"]),
                    node_kind=_required_str(row["node_kind"], default="concept"),
                    source_type=_optional_str(row.get("source_type")),
                    source_id=_optional_str(row.get("source_id")),
                    title=_optional_str(row.get("title")),
                    content=_required_str(row.get("content")),
                    metadata=_to_struct_data(row.get("struct_data")),
                    taxonomy_path=_optional_str(row.get("taxonomy_path")),
                    tenant_id=_optional_str(row.get("tenant_id")),
                    user_id=_optional_str(row.get("user_id")),
                )
            )
        return nodes

    def _fuse_results(
        self,
        *,
        vector_hits: dict[str, float],
        text_hits: dict[str, float],
        fusion: str,
        rrf_k: int,
        vector_weight: float,
        text_weight: float,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Merge vector and text hit maps into a ranked ``[(id, score)]`` list.

        Supports ``"rrf"`` (reciprocal rank fusion) and ``"weighted"`` (linear
        combination using ``vector_weight`` / ``text_weight``).
        """
        ids = set(vector_hits) | set(text_hits)
        if not ids:
            return []
        scored: list[tuple[str, float]]
        if fusion == "rrf":
            vec_rank = {rid: rank for rank, rid in enumerate(vector_hits.keys(), start=1)}
            text_rank = {rid: rank for rank, rid in enumerate(text_hits.keys(), start=1)}
            scored = []
            for rid in ids:
                score = 0.0
                if rid in vec_rank:
                    score += 1.0 / (rrf_k + vec_rank[rid])
                if rid in text_rank:
                    score += 1.0 / (rrf_k + text_rank[rid])
                scored.append((rid, score))
        else:
            scored = [
                (
                    rid,
                    (vector_weight * vector_hits.get(rid, 0.0))
                    + (text_weight * text_hits.get(rid, 0.0)),
                )
                for rid in ids
            ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _build_scope_filters(
        self,
        *,
        tenant_id: str,
        user_id: str | None,
        scope: TaxonomyPath | None,
        source_types: list[str] | None,
    ) -> tuple[list[sql.Composable], list[object]]:
        """Build WHERE clauses for tenant, user, taxonomy scope, and source type.

        Returns a ``(clauses, params)`` pair ready for ``sql.SQL.join``.
        """
        where: list[sql.Composable] = [sql.SQL("tenant_id = %s")]
        params: list[object] = [tenant_id]
        if user_id:
            where.append(sql.SQL("(user_id = %s OR user_id IS NULL)"))
            params.append(user_id)
        if scope:
            where.append(sql.SQL("taxonomy_path <@ %s::ltree"))
            params.append(scope.path)
        if source_types:
            where.append(sql.SQL("source_type = ANY(%s::text[])"))
            params.append(source_types)
        return where, params


__all__ = ["PostgresKnowledgeStore"]
