"""Knowledge graph enricher for text chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from .serialization import load_graph_secure

logger = logging.getLogger(__name__)


class GraphEnricher:
    """Enriches text chunks with knowledge graph context."""

    def __init__(
        self,
        graph_path: Path,
        taxonomy_path: Path | None = None,
    ) -> None:
        """Initialize enricher by loading graph and taxonomy from disk.

        Args:
            graph_path: Path to the saved graph file (pickle format)
            taxonomy_path: Optional path to taxonomy.json for category lookup
        """
        # Load graph
        if not graph_path.exists():
            logger.warning("Graph file not found: %s. Graph enrichment disabled.", graph_path)
            self.graph = nx.Graph()
            self._graph_enabled = False
        else:
            try:
                # Load graph with integrity verification
                self.graph = load_graph_secure(graph_path)
                logger.info(
                    "Loaded graph securely: %d nodes, %d edges",
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
                self._graph_enabled = True
            except Exception as e:
                logger.error(
                    "Failed to load graph from %s: %s. Graph enrichment disabled.",
                    graph_path,
                    e,
                )
                self.graph = nx.Graph()
                self._graph_enabled = False

        # Load taxonomy
        self.taxonomy: dict[str, Any] | None = None
        self._keyword_to_category: dict[str, str] = {}

        if taxonomy_path and taxonomy_path.exists():
            try:
                with open(taxonomy_path, encoding="utf-8") as f:
                    self.taxonomy = json.load(f)
                # Build reverse lookup: keyword -> category
                for cat_name, cat_data in self.taxonomy.get("categories", {}).items():
                    for keyword in cat_data.get("keywords", []):
                        self._keyword_to_category[keyword.lower()] = cat_name
                logger.info(
                    "Loaded taxonomy with %d keyword->category mappings",
                    len(self._keyword_to_category),
                )
            except Exception as e:
                logger.warning("Failed to load taxonomy: %s", e)

    def _get_category_for_entity(self, entity: str) -> str | None:
        """Get parent category for an entity from taxonomy.

        Args:
            entity: Entity name

        Returns:
            Category name or None if not found
        """
        if not self.taxonomy:
            return None

        entity_lower = entity.lower()

        # Direct lookup
        if entity_lower in self._keyword_to_category:
            return self._keyword_to_category[entity_lower]

        # Try canonical map lookup
        canonical_map = self.taxonomy.get("canonical_map", {})
        canonical = canonical_map.get(entity_lower)
        if canonical and canonical.lower() in self._keyword_to_category:
            return self._keyword_to_category[canonical.lower()]

        return None

    def get_context(self, text_chunk: str) -> dict[str, Any]:
        """Get graph context for a text chunk.

        Args:
            text_chunk: The text to enrich

        Returns:
            Dictionary with:
            - 'keywords' (list): Matched entities and their neighbors
            - 'summary' (string): Relation descriptions
            - 'parent_categories' (list): Taxonomy categories for matched entities
        """
        if not self._graph_enabled:
            return {
                "keywords": [],
                "summary": "",
                "parent_categories": [],
            }

        matched_entities: list[str] = []
        neighbors: list[str] = []
        relations: list[str] = []
        parent_categories: set[str] = set()

        text_lower = text_chunk.lower()

        # Optimization: limit search to avoid O(n) scan on large graphs
        # For very short chunks, skip expensive matching
        if len(text_chunk) < 20:
            return {
                "keywords": [],
                "summary": "",
                "parent_categories": [],
            }

        # Entity matching: case-insensitive substring matching
        # NOTE: Worst-case is O(n) over graph nodes. This is acceptable for small graphs,
        # but if the graph grows large, we should build an index (e.g., token->nodes or
        # ngram->nodes) and use that for candidate selection.
        for node in self.graph.nodes():
            node_lower = str(node).lower()
            if node_lower in text_lower or text_lower in node_lower:
                matched_entities.append(str(node))

                # Get parent category from taxonomy
                category = self._get_category_for_entity(str(node))
                if category:
                    parent_categories.add(category)

                # Find 1-hop neighbors
                for neighbor in self.graph.neighbors(node):
                    neighbors.append(str(neighbor))

                    # Get parent category for neighbor too
                    neighbor_category = self._get_category_for_entity(str(neighbor))
                    if neighbor_category:
                        parent_categories.add(neighbor_category)

                    # Get relation label if available
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if edge_data:
                        relation_label = edge_data.get("relation", "")
                        if relation_label:
                            relations.append(f"{node} connects to {neighbor} via {relation_label}")

        # Build keywords list (matched entities + neighbors, deduplicated)
        all_keywords = sorted(set(matched_entities + neighbors))

        # Build summary string
        if relations:
            summary = ". ".join(relations[:5])  # Limit to 5 relations
        else:
            summary = ""

        return {
            "keywords": all_keywords,
            "summary": summary,
            "parent_categories": sorted(parent_categories),
        }
