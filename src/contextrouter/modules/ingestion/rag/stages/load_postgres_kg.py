"""Stage: Load KG nodes/edges/aliases into Postgres."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from psycopg_pool import AsyncConnectionPool

from contextrouter.core import get_core_config
from contextrouter.modules.providers.storage.postgres.models import GraphEdge, GraphNode
from contextrouter.modules.providers.storage.postgres.store import PostgresKnowledgeStore

from ..config import get_assets_paths
from ..settings import RagIngestionConfig

logger = logging.getLogger(__name__)


def load_postgres_kg(*, config: RagIngestionConfig) -> dict[str, str]:
    return asyncio.run(_load_async(config=config))


async def _load_async(*, config: RagIngestionConfig) -> dict[str, str]:
    core_cfg = get_core_config()
    dsn = config.upload.postgres.dsn or core_cfg.postgres.dsn
    if not dsn:
        raise ValueError("Postgres DSN is required for KG load")
    tenant_id = config.upload.postgres.tenant_id or "public"
    user_id = config.upload.postgres.user_id

    paths = get_assets_paths(config)
    nodes_path = paths["assets"] / "knowledge_nodes.jsonl"
    edges_path = paths["assets"] / "knowledge_edges.jsonl"
    aliases_path = paths["assets"] / "knowledge_aliases.jsonl"

    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)

    store = PostgresKnowledgeStore(
        dsn=dsn,
        pool_min_size=config.upload.postgres.pool_min_size,
        pool_max_size=config.upload.postgres.pool_max_size,
    )
    await store.upsert_graph(nodes, edges, tenant_id=str(tenant_id), user_id=user_id)

    if aliases_path.exists():
        await _load_aliases(
            dsn=dsn,
            tenant_id=str(tenant_id),
            aliases=_load_jsonl(aliases_path),
        )

    return {
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "aliases_path": str(aliases_path),
    }


def _load_nodes(path: Path) -> list[GraphNode]:
    rows = _load_jsonl(path)
    nodes: list[GraphNode] = []
    for row in rows:
        nodes.append(
            GraphNode(
                id=str(row.get("id") or ""),
                content=str(row.get("content") or ""),
                node_kind=str(row.get("node_kind") or "concept"),
                source_type=row.get("source_type"),
                source_id=row.get("source_id"),
                title=row.get("title"),
                taxonomy_path=row.get("taxonomy_path"),
                metadata=row.get("struct_data") or {},
            )
        )
    return nodes


def _load_edges(path: Path) -> list[GraphEdge]:
    rows = _load_jsonl(path)
    edges: list[GraphEdge] = []
    for row in rows:
        edges.append(
            GraphEdge(
                source_id=str(row.get("source_id") or ""),
                target_id=str(row.get("target_id") or ""),
                relation=str(row.get("relation") or "relates_to"),
                weight=float(row.get("weight") or 1.0),
                metadata=row.get("metadata") or {},
            )
        )
    return edges


async def _load_aliases(*, dsn: str, tenant_id: str, aliases: list[dict[str, Any]]) -> None:
    pool = AsyncConnectionPool(dsn, min_size=2, max_size=10, open=False)
    if not pool.opened:
        await pool.open()
    async with pool.connection() as conn:
        async with conn.transaction():
            for row in aliases:
                alias = str(row.get("alias") or "").strip()
                node_id = str(row.get("node_id") or "").strip()
                source = str(row.get("source") or "").strip()
                if not alias or not node_id or not source:
                    continue
                await conn.execute(
                    """
                    INSERT INTO knowledge_aliases (tenant_id, alias, node_id, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (tenant_id, alias) DO UPDATE
                    SET node_id = EXCLUDED.node_id,
                        source = EXCLUDED.source
                    """,
                    [tenant_id, alias, node_id, source],
                )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


__all__ = ["load_postgres_kg"]
