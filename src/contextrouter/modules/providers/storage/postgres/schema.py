"""Postgres schema DDL for knowledge store (pgvector + ltree)."""

from __future__ import annotations

from typing import Sequence


def build_schema_sql(*, vector_dim: int) -> Sequence[str]:
    if vector_dim <= 0:
        raise ValueError("vector_dim must be positive")

    return [
        # Extensions
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS ltree;",
        # Nodes table
        """
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            id              TEXT PRIMARY KEY,
            tenant_id       TEXT NOT NULL,
            user_id         TEXT NULL,
            node_kind       TEXT NOT NULL CHECK (node_kind IN ('chunk', 'concept')),

            source_type     TEXT NULL CHECK (source_type IN ('video','book','qa','web','knowledge')),
            source_id       TEXT NULL,
            title           TEXT NULL,
            content         TEXT NOT NULL,
            struct_data     JSONB NOT NULL DEFAULT '{}'::jsonb,
            keywords_text   TEXT NULL,

            content_hash    TEXT NULL,
            taxonomy_path   LTREE NULL,

            search_vector   TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            keywords_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', COALESCE(keywords_text, ''))) STORED,
            embedding       VECTOR(%d) NULL,

            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        % int(vector_dim),
        # Indexes
        """
        CREATE INDEX IF NOT EXISTS knowledge_nodes_taxonomy_path_gist
          ON knowledge_nodes USING GIST (taxonomy_path);
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_nodes_embedding_hnsw
          ON knowledge_nodes USING hnsw (embedding vector_cosine_ops);
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_nodes_search_vector_gin
          ON knowledge_nodes USING GIN (search_vector);
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_nodes_keywords_vector_gin
          ON knowledge_nodes USING GIN (keywords_vector);
        """,
        "CREATE INDEX IF NOT EXISTS knowledge_nodes_source_type_idx ON knowledge_nodes (source_type);",
        "CREATE INDEX IF NOT EXISTS knowledge_nodes_source_id_idx ON knowledge_nodes (source_id);",
        "CREATE INDEX IF NOT EXISTS knowledge_nodes_node_kind_idx ON knowledge_nodes (node_kind);",
        "CREATE INDEX IF NOT EXISTS knowledge_nodes_tenant_idx ON knowledge_nodes (tenant_id);",
        """
        CREATE INDEX IF NOT EXISTS knowledge_nodes_struct_data_gin
          ON knowledge_nodes USING GIN (struct_data);
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS knowledge_nodes_chunk_content_hash_uq
          ON knowledge_nodes (node_kind, content_hash)
          WHERE node_kind = 'chunk' AND content_hash IS NOT NULL;
        """,
        # Edges table
        """
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            tenant_id   TEXT NOT NULL,
            source_id   TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            target_id   TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            relation    TEXT NOT NULL,
            weight      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
            PRIMARY KEY (tenant_id, source_id, target_id, relation)
        );
        """,
        "CREATE INDEX IF NOT EXISTS knowledge_edges_source_idx ON knowledge_edges (source_id);",
        "CREATE INDEX IF NOT EXISTS knowledge_edges_target_idx ON knowledge_edges (target_id);",
        "CREATE INDEX IF NOT EXISTS knowledge_edges_relation_idx ON knowledge_edges (relation);",
        "CREATE INDEX IF NOT EXISTS knowledge_edges_tenant_idx ON knowledge_edges (tenant_id);",
        # Aliases table
        """
        CREATE TABLE IF NOT EXISTS knowledge_aliases (
            tenant_id   TEXT NOT NULL,
            alias       TEXT NOT NULL,
            node_id     TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            source      TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, alias)
        );
        """,
        "CREATE INDEX IF NOT EXISTS knowledge_aliases_node_id_idx ON knowledge_aliases (node_id);",
        "CREATE INDEX IF NOT EXISTS knowledge_aliases_tenant_idx ON knowledge_aliases (tenant_id);",
    ]


__all__ = ["build_schema_sql"]
