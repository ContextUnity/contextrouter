"""Storage providers.

Active providers:
- BrainProvider: delegates to ContextBrain gRPC (primary)
- GCSProvider: Google Cloud Storage read/write
- VertexProvider: Vertex AI Search retrieval
- PostgresProvider: local pgvector hybrid search
"""

from __future__ import annotations

from .brain import BrainProvider
from .gcs import GCSProvider
from .postgres.provider import PostgresProvider
from .vertex import VertexProvider

__all__ = [
    "BrainProvider",
    "GCSProvider",
    "PostgresProvider",
    "VertexProvider",
]
