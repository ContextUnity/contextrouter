"""Graph Manager Service for runtime knowledge graph access.

**Why in `cortex/services/`?**
The graph service is part of the runtime cortex because:
1. It's consumed by brain nodes (intent detection, retrieval) at runtime
2. It's a shared service across multiple nodes, not ingestion-specific
3. It provides runtime knowledge access, not ingestion-time processing
4. The cortex owns runtime knowledge access patterns (graph + taxonomy)
5. Separation: ingestion builds the graph, cortex consumes it at runtime

This module provides a singleton pattern for loading the knowledge graph
once at application startup and providing context lookups for LLM prompts.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from contextrouter.core import get_core_config, get_env

from .local import GraphService
from .postgres import PostgresGraphService

logger = logging.getLogger(__name__)

# Module-level singleton instance and lock
_graph_service: "GraphService | None" = None
_graph_lock = threading.Lock()


def get_graph_service(
    graph_path: Path | None = None,
    taxonomy_path: Path | None = None,
    ontology_path: Path | None = None,
) -> GraphService:
    """Get or create the singleton GraphService instance.

    Thread-safe singleton pattern.

    Args:
        graph_path: Path to knowledge_graph.pickle (only used on first call)
        taxonomy_path: Path to taxonomy.json (only used on first call)
        ontology_path: Path to ontology.json (only used on first call)

    Returns:
        Singleton GraphService instance
    """
    global _graph_service

    if _graph_service is not None:
        return _graph_service

    with _graph_lock:
        # Double-check after acquiring lock
        if _graph_service is not None:
            return _graph_service

        # Determine default paths if not provided
        if graph_path is None or taxonomy_path is None or ontology_path is None:
            try:
                from contextbrain.ingestion.rag import (
                    get_assets_paths,
                    load_config,
                )

                config = load_config()
                paths = get_assets_paths(config)
                if graph_path is None:
                    graph_path = paths.get("graph")
                if taxonomy_path is None:
                    taxonomy_path = paths.get("taxonomy")
                if ontology_path is None:
                    ontology_path = paths.get("ontology")
            except ImportError:
                logger.warning(
                    "Could not load config for default paths (contextbrain not available)"
                )

        # Back-compat: some repos store graph as knowledge_graph.gpickle
        if graph_path is not None and not graph_path.exists():
            alt = graph_path.with_name("knowledge_graph.gpickle")
            if alt.exists():
                graph_path = alt

        cfg = get_core_config()
        kg_backend = (get_env("RAG_KG_BACKEND") or "").strip().lower()
        if kg_backend == "postgres" and cfg.postgres.dsn:
            tenant_id = get_env("RAG_TENANT_ID") or "public"
            _graph_service = PostgresGraphService(
                dsn=cfg.postgres.dsn,
                tenant_id=str(tenant_id),
                taxonomy_path=taxonomy_path,
                ontology_path=ontology_path,
            )
        else:
            _graph_service = GraphService(
                graph_path=graph_path,
                taxonomy_path=taxonomy_path,
                ontology_path=ontology_path,
            )

    return _graph_service


def reset_graph_service() -> None:
    """Reset the singleton (mainly for testing)."""
    global _graph_service
    with _graph_lock:
        _graph_service = None


# Re-export all public API
__all__ = [
    "GraphService",
    "PostgresGraphService",
    "get_graph_service",
    "reset_graph_service",
]
