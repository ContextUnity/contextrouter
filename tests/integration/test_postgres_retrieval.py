from __future__ import annotations

import os

import pytest
from psycopg_pool import AsyncConnectionPool

from contextrouter.modules.providers.storage.postgres.schema import build_schema_sql
from contextrouter.modules.providers.storage.postgres.store import PostgresKnowledgeStore


@pytest.mark.asyncio
async def test_postgres_hybrid_search_vector():
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")

    pool = AsyncConnectionPool(dsn, open=False)
    await pool.open()
    async with pool.connection() as conn:
        for stmt in build_schema_sql(vector_dim=3):
            await conn.execute(stmt)
        await conn.execute("DELETE FROM knowledge_nodes WHERE tenant_id = %s", ["test"])
        await conn.execute(
            """
            INSERT INTO knowledge_nodes (id, tenant_id, node_kind, content, struct_data, embedding)
            VALUES (%s, %s, 'chunk', %s, %s, %s::vector),
                   (%s, %s, 'chunk', %s, %s, %s::vector)
            """,
            [
                "n1",
                "test",
                "alpha beta",
                {},
                "[0.1,0.1,0.1]",
                "n2",
                "test",
                "gamma delta",
                {},
                "[0.9,0.9,0.9]",
            ],
        )

    store = PostgresKnowledgeStore(dsn=dsn)
    results = await store.hybrid_search(
        query_text="",
        query_vec=[0.1, 0.1, 0.1],
        candidate_k=2,
        limit=1,
        tenant_id="test",
        fusion="weighted",
        vector_weight=1.0,
        text_weight=0.0,
    )
    assert results
    assert results[0].node.id == "n1"


@pytest.mark.asyncio
async def test_postgres_poison_pill_embedding_dimension():
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")

    pool = AsyncConnectionPool(dsn, open=False)
    await pool.open()
    async with pool.connection() as conn:
        for stmt in build_schema_sql(vector_dim=3):
            await conn.execute(stmt)
        await conn.execute("DELETE FROM knowledge_nodes WHERE tenant_id = %s", ["test"])

        with pytest.raises(Exception):
            await conn.execute(
                """
                INSERT INTO knowledge_nodes (id, tenant_id, node_kind, content, struct_data, embedding)
                VALUES (%s, %s, 'chunk', %s, %s, %s::vector)
                """,
                ["bad", "test", "poison", {}, "[0.1,0.2]"],
            )
