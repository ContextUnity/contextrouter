"""PostgresGraphService - Postgres-backed knowledge graph access.

Uses PostgreSQL for graph storage instead of local pickle files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg_pool import ConnectionPool

from .local import GraphService

logger = logging.getLogger(__name__)


class PostgresGraphService(GraphService):
    """Postgres-backed KG facts lookup (no local pickle)."""

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
        super().__init__(graph_path=None, taxonomy_path=taxonomy_path, ontology_path=ontology_path)
        self._pool = ConnectionPool(dsn, min_size=1, max_size=5)
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._max_hops = max(1, int(max_hops))
        self._max_facts = max(1, int(max_facts))

    def get_facts(self, concepts: list[str]) -> list[str]:
        """Get facts from Postgres knowledge graph."""
        if not concepts:
            return []
        if not self._tenant_id:
            return []

        concept_keys = [c.strip().lower() for c in concepts if isinstance(c, str) and c.strip()]
        if not concept_keys:
            return []

        allowed = sorted(self._fact_labels) if self._fact_labels else None
        with self._pool.connection() as conn:
            entrypoints = self._resolve_entrypoints(conn, concept_keys)
            if not entrypoints:
                return []
            rows = self._fetch_facts(conn, entrypoints=entrypoints, allowed_relations=allowed)
            facts: list[str] = []
            seen: set[tuple[str, str, str]] = set()
            for row in rows:
                src = str(row.get("source_content") or "").strip()
                tgt = str(row.get("target_content") or "").strip()
                rel = str(row.get("relation") or "").strip()
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

    def _resolve_entrypoints(self, conn, concept_keys: list[str]) -> list[str]:
        """Resolve concept keys to node IDs via aliases or direct lookup."""
        rows = conn.execute(
            """
            SELECT node_id
            FROM knowledge_aliases
            WHERE tenant_id = %s
              AND alias = ANY(%s::text[])
            """,
            [self._tenant_id, concept_keys],
        )
        entrypoints = [r[0] for r in rows.fetchall()]
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
            [self._tenant_id, concept_keys],
        )
        return [r[0] for r in rows.fetchall()]

    def _fetch_facts(self, conn, *, entrypoints: list[str], allowed_relations: list[str] | None):
        """Fetch facts via recursive CTE traversal."""
        return conn.execute(
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
                self._tenant_id,
                entrypoints,
                allowed_relations,
                allowed_relations,
                self._max_hops,
                self._tenant_id,
                allowed_relations,
                allowed_relations,
                self._max_facts,
            ],
        )


__all__ = ["PostgresGraphService"]
