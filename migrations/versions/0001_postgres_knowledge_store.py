"""Create knowledge store tables."""

from __future__ import annotations

from alembic import op

from contextrouter.core.config import get_env
from contextrouter.modules.providers.storage.postgres.schema import build_schema_sql

# revision identifiers, used by Alembic.
revision = "0001_postgres_knowledge_store"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    vector_dim = int(get_env("PGVECTOR_DIM") or 768)
    for stmt in build_schema_sql(vector_dim=vector_dim):
        op.execute(stmt)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledge_aliases;")
    op.execute("DROP TABLE IF EXISTS knowledge_edges;")
    op.execute("DROP TABLE IF EXISTS knowledge_nodes;")
