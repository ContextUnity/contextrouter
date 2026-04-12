"""Knowledge graph queries (Postgres)."""

from __future__ import annotations

from typing import Iterable

from psycopg.rows import dict_row


async def fetch_kg_facts(
    *,
    conn,
    tenant_id: str,
    entrypoints: Iterable[str],
    allowed_relations: list[str] | None,
    max_depth: int,
    max_facts: int,
) -> list[tuple[str, str, str]]:
    if not entrypoints:
        return []
    conn.row_factory = dict_row
    sql = """
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
    SELECT source_id, target_id, relation
    FROM walk
    LIMIT %s
    """
    params = [
        tenant_id,
        list(entrypoints),
        allowed_relations,
        allowed_relations,
        max_depth,
        tenant_id,
        allowed_relations,
        allowed_relations,
        max_facts,
    ]
    rows = await conn.execute(sql, params)
    out: list[tuple[str, str, str]] = []
    async for row in rows:
        out.append((row["source_id"], row["target_id"], row["relation"]))
    return out


__all__ = ["fetch_kg_facts"]
