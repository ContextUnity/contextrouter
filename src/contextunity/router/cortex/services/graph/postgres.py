"""PostgresGraphService - Postgres-backed knowledge graph access.
Uses PostgreSQL for graph storage instead of local pickle files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeGuard, override, runtime_checkable

from contextunity.core import get_contextunit_logger
from psycopg_pool import ConnectionPool

from .local import GraphService

logger = get_contextunit_logger(__name__)


class _PostgresCursorLike(Protocol):
    """Typed fetch boundary for psycopg cursors."""

    def fetchone(self) -> object: ...


@runtime_checkable
class _PostgresFetchOneMethod(Protocol):
    def __call__(self) -> object: ...


@runtime_checkable
class _PostgresExecuteMethod(Protocol):
    def __call__(self, query: str, params: list[object], /) -> object: ...


class _PostgresCursorAdapter:
    """Typed cursor adapter over psycopg row fetches."""

    _cursor_obj: object

    def __init__(self, cursor_obj: object) -> None:
        self._cursor_obj = cursor_obj

    def fetchone(self) -> object:
        method = getattr(self._cursor_obj, "fetchone", None)
        if not isinstance(method, _PostgresFetchOneMethod):
            raise TypeError("Postgres cursor missing fetchone")
        return method()


class _PostgresConnectionAdapter:
    """Typed connection adapter over psycopg execute calls."""

    _conn_obj: object

    def __init__(self, conn_obj: object) -> None:
        self._conn_obj = conn_obj

    def execute(self, query: str, params: list[object], /) -> _PostgresCursorLike:
        method = getattr(self._conn_obj, "execute", None)
        if not isinstance(method, _PostgresExecuteMethod):
            raise TypeError("Postgres connection missing execute")
        return _PostgresCursorAdapter(method(query, params))


def _fetchone_row(cursor: _PostgresCursorLike) -> object:
    """Fetch a single untrusted Postgres row."""
    return cursor.fetchone()


def _is_single_text_row(value: object) -> TypeGuard[tuple[str]]:
    """Validate ``SELECT single_text_column`` postgres row."""
    match value:
        case (str(),):
            return True
        case _:
            return False


def _is_fact_row(value: object) -> TypeGuard[tuple[object, object, str, str, str]]:
    """Validate graph fact row: ``(source_id, target_id, relation, source_content, target_content)``."""
    match value:
        case (object(), object(), str(), str(), str()):
            return True
        case _:
            return False


class PostgresGraphService(GraphService):
    """PostgreSQL-backed knowledge graph service.

    Replaces local pickle-based graph storage with SQL-driven fact retrieval
    using recursive CTE traversal. Inherits taxonomy and ontology loading
    from the base GraphService.
    """

    def __init__(
        self,
        *,
        dsn: str,
        tenant_id: str,
        user_id: str | None = None,
        taxonomy_path: Path | None = None,
        ontology_path: Path | None = None,
        max_hops: int = 2,
        max_facts: int = 30,
    ) -> None:
        """Initialize the PostgreSQL graph service.

        Args:
            dsn: PostgreSQL connection string.
            tenant_id: Tenant identifier for row-level isolation.
            user_id: Optional user identifier for per-user graph scoping.
            taxonomy_path: Path to taxonomy.json for keyword-category lookup.
            ontology_path: Path to ontology.json for relation label filtering.
            max_hops: Maximum depth for recursive edge traversal (default 2).
            max_facts: Maximum number of fact strings to return (default 30).
        """
        super().__init__(graph_path=None, taxonomy_path=taxonomy_path, ontology_path=ontology_path)
        self._pool: ConnectionPool = ConnectionPool(dsn, min_size=1, max_size=5)
        self._tenant_id: str = tenant_id
        self._user_id: str | None = user_id
        self._max_hops: int = max(1, int(max_hops))
        self._max_facts: int = max(1, int(max_facts))

    def _active_tenant_id(self) -> str:
        """Resolve tenant from request context, falling back to ctor default."""
        from contextunity.router.cortex.services.graph.tenant_context import (
            get_request_tenant_id,
        )

        return get_request_tenant_id() or self._tenant_id

    @override
    def get_facts(self, concepts: list[str]) -> list[str]:
        """Retrieve relationship facts from the PostgreSQL knowledge graph.

        Resolves concept strings to node IDs via alias or direct lookup,
        then traverses edges up to ``max_hops`` deep using a recursive CTE.

        Args:
            concepts: List of concept strings to look up.

        Returns:
            A list of human-readable fact strings (e.g., "Fact: Fear causes Poverty").
        """
        if not concepts:
            return []
        tenant_id = self._active_tenant_id()
        if not tenant_id:
            return []

        concept_keys = [c.strip().lower() for c in concepts if c.strip()]
        if not concept_keys:
            return []

        allowed = sorted(self._fact_labels) if self._fact_labels else None
        with self._pool.connection() as raw_conn:
            conn = _PostgresConnectionAdapter(raw_conn)
            entrypoints = self._resolve_entrypoints(conn, concept_keys, tenant_id=tenant_id)
            if not entrypoints:
                return []
            facts: list[str] = []
            seen: set[tuple[str, str, str]] = set()
            rows = self._fetch_facts(
                conn,
                entrypoints=entrypoints,
                allowed_relations=allowed,
                tenant_id=tenant_id,
            )
            for row in rows:
                _source_id, _target_id, relation, src, tgt = row
                src = src.strip()
                tgt = tgt.strip()
                rel = relation.strip()
                if not src or not tgt or not rel:
                    continue
                key = (src.lower(), rel.lower(), tgt.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(f"Fact: {src} {rel} {tgt}")
                if len(facts) >= self._max_facts:
                    break
            return facts

    def _resolve_entrypoints(
        self,
        conn: _PostgresConnectionAdapter,
        concept_keys: list[str],
        *,
        tenant_id: str,
    ) -> list[str]:
        """Resolve lowercased concept keys to database node IDs.

        First checks the ``knowledge_aliases`` table, then falls back to
        direct content matching in ``knowledge_nodes``.

        Args:
            conn: Active psycopg connection from the pool.
            concept_keys: Lowercased and stripped concept strings.

        Returns:
            A list of node ID strings, possibly empty.
        """
        rows = conn.execute(
            """
            SELECT node_id
            FROM knowledge_aliases
            WHERE tenant_id = %s
              AND alias = ANY(%s::text[])
            """,
            [tenant_id, concept_keys],
        )
        entrypoints: list[str] = []
        while True:
            row = _fetchone_row(rows)
            if row is None:
                break
            if _is_single_text_row(row):
                entrypoints.append(row[0])
        if entrypoints:
            return entrypoints

        rows = conn.execute(
            """
            SELECT id
            FROM knowledge_nodes
            WHERE tenant_id = %s
              AND node_kind = 'concept'
              AND lower(content) = ANY(%s::text[])
            """,
            [tenant_id, concept_keys],
        )
        direct_entrypoints: list[str] = []
        while True:
            row = _fetchone_row(rows)
            if row is None:
                break
            if _is_single_text_row(row):
                direct_entrypoints.append(row[0])
        return direct_entrypoints

    def _fetch_facts(
        self,
        conn: _PostgresConnectionAdapter,
        *,
        entrypoints: list[str],
        allowed_relations: list[str] | None,
        tenant_id: str,
    ) -> list[tuple[object, object, str, str, str]]:
        """Execute a recursive CTE traversal to fetch edge facts.

        Walks edges from entrypoint nodes up to ``max_hops`` levels deep,
        optionally filtering by allowed relation labels from the ontology.

        Args:
            conn: Active psycopg connection from the pool.
            entrypoints: Starting node IDs for the traversal.
            allowed_relations: Optional relation label allowlist. None means all relations.

        Returns:
            Typed row tuples containing source/target ids, relation, and content strings.
        """
        rows = conn.execute(
            """
            WITH RECURSIVE walk AS (
                SELECT source_id, target_id, relation, 1 AS depth
                FROM knowledge_edges
                WHERE tenant_id = %s
                  AND source_id = ANY(%s::text[])
                  AND (%s::text[] IS NULL OR relation = ANY(%s::text[]))
                UNION ALL
                SELECT e.source_id, e.target_id, e.relation, w.depth + 1
                FROM walk w
                JOIN knowledge_edges e ON e.source_id = w.target_id
                WHERE w.depth < %s
                  AND e.tenant_id = %s
                  AND (%s::text[] IS NULL OR e.relation = ANY(%s::text[]))
            )
            SELECT w.source_id, w.target_id, w.relation,
                   ns.content AS source_content, nt.content AS target_content
            FROM walk w
            JOIN knowledge_nodes ns ON ns.id = w.source_id
            JOIN knowledge_nodes nt ON nt.id = w.target_id
            LIMIT %s
            """,
            [
                tenant_id,
                entrypoints,
                allowed_relations,
                allowed_relations,
                self._max_hops,
                tenant_id,
                allowed_relations,
                allowed_relations,
                self._max_facts,
            ],
        )
        facts: list[tuple[object, object, str, str, str]] = []
        while True:
            row = _fetchone_row(rows)
            if row is None:
                break
            if _is_fact_row(row):
                facts.append(row)
        return facts


__all__ = ["PostgresGraphService"]
