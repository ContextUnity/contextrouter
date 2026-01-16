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

The graph context is NOT a citation - it's a "knowledge hint" added to
provide explicit relationships that vector search might miss.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from psycopg_pool import ConnectionPool

from contextrouter.core import get_core_config, get_env
from contextrouter.modules.ingestion.rag.graph.serialization import load_graph_secure

logger = logging.getLogger(__name__)

# Module-level singleton instance and lock
_graph_service: "GraphService | None" = None
_graph_lock = threading.Lock()


class GraphService:
    """Singleton service for knowledge graph access at runtime.

    Thread-safe implementation using module-level lock.
    Degrades gracefully if graph file is missing.
    """

    def __init__(
        self,
        graph_path: Path | None = None,
        taxonomy_path: Path | None = None,
        ontology_path: Path | None = None,
    ) -> None:
        """Initialize graph service.

        Args:
            graph_path: Path to knowledge_graph.pickle
            taxonomy_path: Path to taxonomy.json
        """
        import networkx as nx

        self.graph: nx.Graph = nx.Graph()
        self.taxonomy: dict[str, object] = {}
        self._keyword_to_category: dict[str, str] = {}
        self._node_index: dict[str, object] = {}
        self.ontology: dict[str, object] = {}
        self._fact_labels: set[str] = set()
        self._graph_enabled = False
        self._taxonomy_enabled = False
        self._ontology_enabled = False

        # Load graph securely
        if graph_path and graph_path.exists():
            try:
                # Load graph with integrity verification
                self.graph = load_graph_secure(graph_path)

                # Build a fast case-insensitive node index for O(1) lookups.
                self._node_index = {}
                try:
                    for node in self.graph.nodes():
                        k = str(node).strip().lower()
                        if k and k not in self._node_index:
                            self._node_index[k] = node
                except Exception:
                    self._node_index = {}
                logger.info(
                    "GraphService loaded graph: %d nodes, %d edges",
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
                self._graph_enabled = True
            except Exception as e:
                logger.warning("Failed to load graph: %s. Graph service disabled.", e)
                self._graph_enabled = False
                self._node_index = {}
                self.graph = None
        else:
            logger.warning("Graph file not found: %s. Graph service disabled.", graph_path)
            self._graph_enabled = False
            self._node_index = {}

        # Load taxonomy
        if taxonomy_path and taxonomy_path.exists():
            try:
                with open(taxonomy_path, encoding="utf-8") as f:
                    raw_taxonomy = json.load(f)
                    self.taxonomy = raw_taxonomy if isinstance(raw_taxonomy, dict) else {}
                # Build keyword->category lookup
                cats = self.taxonomy.get("categories")
                if isinstance(cats, dict):
                    for cat_name, cat_data in cats.items():
                        if not isinstance(cat_name, str) or not isinstance(cat_data, dict):
                            continue
                        keywords = cat_data.get("keywords")
                        if isinstance(keywords, list):
                            for keyword in keywords:
                                if isinstance(keyword, str) and keyword.strip():
                                    self._keyword_to_category[keyword.strip().lower()] = cat_name
                logger.info(
                    "GraphService loaded taxonomy: %d categories, %d keywords",
                    len(cats) if isinstance(cats, dict) else 0,
                    len(self._keyword_to_category),
                )
                self._taxonomy_enabled = True
            except Exception as e:
                logger.warning("Failed to load taxonomy: %s. Taxonomy features disabled.", e)
                self._taxonomy_enabled = False
        else:
            logger.warning(
                "Taxonomy file not found: %s. Taxonomy features disabled.",
                taxonomy_path,
            )
            self._taxonomy_enabled = False

        # Load ontology (optional)
        if ontology_path and ontology_path.exists():
            try:
                with open(ontology_path, encoding="utf-8") as f:
                    raw_ontology = json.load(f)
                    self.ontology = raw_ontology if isinstance(raw_ontology, dict) else {}
                rel = self.ontology.get("relations") if isinstance(self.ontology, dict) else None
                rel = rel if isinstance(rel, dict) else {}
                labels = rel.get("runtime_fact_labels") if isinstance(rel, dict) else None
                if isinstance(labels, list):
                    self._fact_labels = {
                        str(x).strip() for x in labels if isinstance(x, str) and str(x).strip()
                    }
                self._ontology_enabled = True
                logger.info(
                    "GraphService loaded ontology: fact_labels=%d",
                    len(self._fact_labels),
                )
            except Exception as e:
                logger.warning("Failed to load ontology: %s. Ontology features disabled.", e)
                self._ontology_enabled = False
                self._fact_labels = set()
        else:
            self._ontology_enabled = False
            self._fact_labels = set()

    def get_context(self, concept: str) -> str:
        """Get graph context for a single concept.

        Returns a compact neighbor/edge summary suitable for model context.

        Args:
            concept: Concept/entity name to look up

        Returns:
            Text summary of relationships, or empty string if not found

        Example output:
            "Concept 'Fear' is linked to 'Poverty' (causes) and 'Faith' (opposes)."
        """
        if not self._graph_enabled or not concept or self.graph.number_of_nodes() == 0:
            return ""

        matching_node = self._node_index.get(concept.strip().lower())
        if matching_node is None:
            return ""

        # Get neighbors and their relationships
        neighbors = list(self.graph.neighbors(matching_node))
        if not neighbors:
            return ""

        # Build relationship descriptions
        relations: list[str] = []
        for neighbor in neighbors[:5]:  # Limit to 5 neighbors
            edge_data = self.graph.get_edge_data(matching_node, neighbor)
            relation = edge_data.get("relation", "relates to") if edge_data else "relates to"
            relations.append(f"'{neighbor}' ({relation})")

        if relations:
            return f"Concept '{matching_node}' is linked to {', '.join(relations)}."

        return ""

    def get_context_for_concepts(self, concepts: list[str]) -> str:
        """Get graph context for multiple concepts (batch, de-duped).

        Args:
            concepts: List of concept/entity names to look up

        Returns:
            Combined text summary of all relationships
        """
        if not self._graph_enabled or not concepts:
            return ""

        # De-duplicate and normalize
        seen: set[str] = set()
        unique_concepts = []
        for c in concepts:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_concepts.append(c)

        # Get context for each concept
        contexts: list[str] = []
        for concept in unique_concepts[:10]:  # Limit to 10 concepts
            ctx = self.get_context(concept)
            if ctx:
                contexts.append(ctx)

        return " ".join(contexts)

    def get_facts(self, concepts: list[str]) -> list[str]:
        """Get explicit relationship facts for a list of concepts.

        This is used for the graph retrieval branch (Path B). Facts are NOT
        citations; they are background knowledge for reasoning.

        Format:
            "Fact: Fear causes Poverty"
        """
        if not self._graph_enabled or not concepts:
            return []

        # De-duplicate input concepts (case-insensitive)
        seen_in: set[str] = set()
        unique: list[str] = []
        for c in concepts:
            if not isinstance(c, str):
                continue
            s = c.strip()
            if not s:
                continue
            k = s.lower()
            if k in seen_in:
                continue
            seen_in.add(k)
            unique.append(s)

        facts: list[str] = []
        seen_fact_keys: set[tuple[str, str, str]] = set()

        for concept in unique[:10]:
            node_match = self._node_index.get(concept.strip().lower())
            if node_match is None:
                continue

            # Emit neighbor facts
            for neighbor in list(self.graph.neighbors(node_match))[:8]:
                edge_data = self.graph.get_edge_data(node_match, neighbor) or {}
                relation = str(edge_data.get("relation") or "relates to").strip()
                if (
                    self._ontology_enabled
                    and self._fact_labels
                    and relation not in self._fact_labels
                ):
                    continue
                src = str(node_match).strip()
                tgt = str(neighbor).strip()
                if not src or not tgt:
                    continue
                key = (src.lower(), relation.lower(), tgt.lower())
                if key in seen_fact_keys:
                    continue
                seen_fact_keys.add(key)
                facts.append(f"Fact: {src} {relation} {tgt}")
                if len(facts) >= 30:
                    return facts

        return facts

    def get_category_for_concept(self, concept: str) -> str | None:
        """Get taxonomy category for a concept.

        Args:
            concept: Concept name

        Returns:
            Category name or None if not found
        """
        if not self._taxonomy_enabled or not concept:
            return None

        concept_lower = concept.lower()

        # Direct lookup
        if concept_lower in self._keyword_to_category:
            return self._keyword_to_category[concept_lower]

        # Try canonical map
        canonical_map = self.taxonomy.get("canonical_map", {})
        canonical = canonical_map.get(concept_lower)
        if canonical and canonical.lower() in self._keyword_to_category:
            return self._keyword_to_category[canonical.lower()]

        return None

    def get_all_categories(self) -> list[str]:
        """Get list of all taxonomy category names.

        Returns:
            List of category names (snake_case)
        """
        if not self._taxonomy_enabled:
            return []
        return list(self.taxonomy.get("categories", {}).keys())

    def get_canonical_map(self) -> dict[str, str]:
        """Get synonym -> canonical term mapping.

        Returns:
            Dictionary mapping synonyms to canonical terms
        """
        if not self._taxonomy_enabled:
            return {}
        return self.taxonomy.get("canonical_map", {})


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
                from contextrouter.modules.ingestion.rag import (
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
                logger.warning("Could not load config for default paths")

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
