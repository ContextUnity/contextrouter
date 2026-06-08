"""GraphService - file-based knowledge graph access.

Uses pickle/joblib files for local graph storage.
Thread-safe with graceful degradation.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import networkx as nx
from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.types import JsonDict, is_json_dict, is_object_dict

from contextunity.router.cortex.types import (
    OntologyData,
    OntologyRelations,
    TaxonomyCategoryData,
    TaxonomyData,
)


class _JoblibLoadModule(Protocol):
    """Typed boundary for joblib's untyped ``load`` helper."""

    def load(self, filename: Path, /) -> object: ...


class _BrainGraphLoader(Protocol):
    """Typed boundary for the optional brain graph loader."""

    def __call__(self, file_path: Path, hash_file_path: Path | None = None, /) -> object: ...


@runtime_checkable
class _GraphNodesMethod(Protocol):
    def __call__(self) -> Iterable[object]: ...


@runtime_checkable
class _GraphEdgesMethod(Protocol):
    def __call__(self, *, data: bool = False) -> Iterable[tuple[object, object, object]]: ...


def _brain_load_graph_candidate(
    loader: _BrainGraphLoader,
    file_path: Path,
    hash_file_path: Path | None,
) -> object:
    """Call the optional brain graph loader through a typed boundary."""
    return loader(file_path, hash_file_path)


def _load_brain_graph_loader() -> _BrainGraphLoader:
    """Resolve the optional Brain loader through an object-typed boundary."""
    module = importlib.import_module("contextunity.brain.ingestion.rag.graph.serialization")
    candidate: object = getattr(module, "load_graph_secure", None)
    if not callable(candidate):
        raise ImportError("Brain graph loader is unavailable")

    def loader(file_path: Path, hash_file_path: Path | None = None, /) -> object:
        return candidate(file_path, hash_file_path)

    return loader


def _joblib_load_candidate(loader: _JoblibLoadModule, file_path: Path) -> object:
    """Call joblib.load through a typed boundary."""
    return loader.load(file_path)


def _normalize_graph(graph_obj: object) -> nx.Graph[str]:
    """Normalize an arbitrary NetworkX graph into ``nx.Graph[str]``."""
    nodes_method = getattr(graph_obj, "nodes", None)
    edges_method = getattr(graph_obj, "edges", None)
    if not isinstance(nodes_method, _GraphNodesMethod) or not isinstance(
        edges_method, _GraphEdgesMethod
    ):
        return nx.Graph[str]()

    normalized = nx.Graph[str]()
    for node in nodes_method():
        normalized.add_node(str(node))
    for source, target, edge_data in edges_method(data=True):
        attrs: dict[str, object] = {}
        if is_object_dict(edge_data):
            relation = edge_data.get("relation")
            if isinstance(relation, str) and relation.strip():
                attrs["relation"] = relation.strip()
        _ = normalized.add_edge(str(source), str(target), **attrs)
    return normalized


def _edge_data_object(graph: nx.Graph[str], source: str, target: str) -> object:
    """Read edge data through an object-typed boundary."""
    edge_data: object = graph.get_edge_data(source, target)
    return edge_data


try:
    _brain_graph_loader = _load_brain_graph_loader()

    def load_graph_secure(file_path: Path, hash_file_path: Path | None = None) -> nx.Graph[str]:
        """Typed wrapper around the optional brain graph loader."""
        loaded_graph = _brain_load_graph_candidate(_brain_graph_loader, file_path, hash_file_path)
        return _normalize_graph(loaded_graph)
except ImportError:

    def load_graph_secure(file_path: Path, hash_file_path: Path | None = None) -> nx.Graph[str]:
        """Fallback implementation if contextunity.brain is not available."""
        _ = hash_file_path
        joblib_loader: _JoblibLoadModule = joblib
        loaded = _joblib_load_candidate(joblib_loader, file_path)
        return _normalize_graph(loaded)


logger = get_contextunit_logger(__name__)

# -- Type aliases --------------------------------------------------------------

"""Keyword → taxonomy category mapping for fast lookup."""
KeywordIndex = dict[str, str]

"""Node label (lowered) → original node id for graph traversal."""
NodeIndex = dict[str, str]


def _load_json_object(path: Path) -> JsonDict | None:
    """Read a JSON file and validate that it is an object."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    decoded = json_loads(raw_text)
    if is_json_dict(decoded):
        return decoded
    return None


def _taxonomy_from_json(data: JsonDict) -> TaxonomyData:
    """Convert a validated JSON object into typed taxonomy data."""
    taxonomy: TaxonomyData = {}

    categories_raw = data.get("categories")
    if isinstance(categories_raw, dict):
        categories: dict[str, TaxonomyCategoryData] = {}
        for name, raw_category in categories_raw.items():
            if not isinstance(raw_category, dict):
                continue
            keywords_raw = raw_category.get("keywords")
            keywords: list[str] = []
            if isinstance(keywords_raw, list):
                for keyword in keywords_raw:
                    if isinstance(keyword, str) and keyword.strip():
                        keywords.append(keyword.strip())
            categories[name] = {"keywords": keywords}
        taxonomy["categories"] = categories

    canonical_map_raw = data.get("canonical_map")
    if isinstance(canonical_map_raw, dict):
        canonical_map: dict[str, str] = {}
        for alias, canonical in canonical_map_raw.items():
            if isinstance(canonical, str):
                canonical_map[alias] = canonical
        taxonomy["canonical_map"] = canonical_map

    return taxonomy


def _ontology_from_json(data: JsonDict) -> OntologyData:
    """Convert a validated JSON object into typed ontology data."""
    ontology: OntologyData = {}
    relations_raw = data.get("relations")
    if isinstance(relations_raw, dict):
        runtime_fact_labels_raw = relations_raw.get("runtime_fact_labels")
        runtime_fact_labels: list[str] = []
        if isinstance(runtime_fact_labels_raw, list):
            for label in runtime_fact_labels_raw:
                if isinstance(label, str) and label.strip():
                    runtime_fact_labels.append(label.strip())
        relations: OntologyRelations = {"runtime_fact_labels": runtime_fact_labels}
        ontology["relations"] = relations
    return ontology


def _edge_relation(graph: nx.Graph[str], source: str, target: str) -> str:
    """Read an edge relation from NetworkX data without leaking untyped values."""
    edge_data = _edge_data_object(graph, source, target)
    if is_object_dict(edge_data):
        relation = edge_data.get("relation")
        if isinstance(relation, str) and relation.strip():
            return relation.strip()
    return "relates to"


# -- Service -------------------------------------------------------------------


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
            ontology_path: Path to ontology.json
        """
        self.graph: nx.Graph[str] = nx.Graph()
        self.taxonomy: TaxonomyData = {}
        self._keyword_to_category: KeywordIndex = {}
        self._node_index: NodeIndex = {}
        self.ontology: OntologyData = {}
        self._fact_labels: set[str] = set()
        self._graph_enabled: bool = False
        self._taxonomy_enabled: bool = False
        self._ontology_enabled: bool = False

        self._load_graph(graph_path)
        self._load_taxonomy(taxonomy_path)
        self._load_ontology(ontology_path)

    def _load_graph(self, graph_path: Path | None) -> None:
        """Load a NetworkX graph from a pickle or joblib file.

        Builds an internal node index (lowered label → original node ID)
        for fast concept lookups. Disables graph features gracefully if
        the file is missing or unreadable.

        Args:
            graph_path: Path to the serialized graph file, or None to skip.
        """
        if graph_path and graph_path.exists():
            try:
                self.graph = load_graph_secure(graph_path)
                self._node_index = {}
                try:
                    for node in self.graph.nodes():
                        k = str(node).strip().lower()
                        if k and k not in self._node_index:
                            self._node_index[k] = node
                except Exception as e:  # graceful-degrade: index build is optional
                    logger.warning(
                        (
                            "GraphService: node-index build failed: %s. "
                            "Concept lookups will be unavailable."
                        ),
                        e,
                    )
                    self._node_index = {}
                logger.info(
                    "GraphService loaded graph: %d nodes, %d edges",
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
                self._graph_enabled = True
            except Exception as e:  # graceful-degrade: missing graph file disables graph features
                logger.warning("Failed to load graph: %s. Graph service disabled.", e)
                self._graph_enabled = False
                self._node_index = {}
                self.graph = nx.Graph()
        else:
            logger.warning("Graph file not found: %s. Graph service disabled.", graph_path)
            self._graph_enabled = False
            self._node_index = {}

    def _load_taxonomy(self, taxonomy_path: Path | None) -> None:
        """Load taxonomy from a JSON file and build a keyword-to-category index.

        Args:
            taxonomy_path: Path to taxonomy.json, or None to skip.
        """
        if taxonomy_path and taxonomy_path.exists():
            try:
                taxonomy_data = _load_json_object(taxonomy_path)
                self.taxonomy = _taxonomy_from_json(taxonomy_data) if taxonomy_data else {}
                cats = self.taxonomy.get("categories", {})
                for cat_name, cat_data in cats.items():
                    for keyword in cat_data.get("keywords", []):
                        self._keyword_to_category[keyword.lower()] = cat_name
                logger.info(
                    "GraphService loaded taxonomy: %d categories, %d keywords",
                    len(cats),
                    len(self._keyword_to_category),
                )
                self._taxonomy_enabled = True
            except Exception as e:  # graceful-degrade: missing taxonomy disables taxonomy features
                logger.warning("Failed to load taxonomy: %s. Taxonomy features disabled.", e)
                self._taxonomy_enabled = False
        else:
            logger.warning(
                "Taxonomy file not found: %s. Taxonomy features disabled.",
                taxonomy_path,
            )
            self._taxonomy_enabled = False

    def _load_ontology(self, ontology_path: Path | None) -> None:
        """Load ontology from a JSON file and extract runtime fact labels.

        Fact labels are used to filter graph edges during ``get_facts()`` calls.

        Args:
            ontology_path: Path to ontology.json, or None to skip.
        """
        if ontology_path and ontology_path.exists():
            try:
                ontology_data = _load_json_object(ontology_path)
                self.ontology = _ontology_from_json(ontology_data) if ontology_data else {}
                rel = self.ontology.get("relations", {})
                labels = rel.get("runtime_fact_labels", [])
                self._fact_labels = {label for label in labels if label}
                self._ontology_enabled = True
                logger.info(
                    "GraphService loaded ontology: fact_labels=%d",
                    len(self._fact_labels),
                )
            except Exception as e:  # graceful-degrade: missing ontology disables ontology features
                logger.warning("Failed to load ontology: %s. Ontology features disabled.", e)
                self._ontology_enabled = False
                self._fact_labels = set()
        else:
            self._ontology_enabled = False
            self._fact_labels = set()

    def get_context(self, concept: str) -> str:
        """Get a compact graph context string for a single concept.

        Looks up the concept in the node index and returns a one-line summary
        of its neighbors and edge relations, suitable for injecting into
        an LLM prompt.

        Args:
            concept: The concept label to look up (case-insensitive).

        Returns:
            A human-readable neighbor summary, or an empty string if not found.
        """
        if not self._graph_enabled or not concept or self.graph.number_of_nodes() == 0:
            return ""

        matching_node = self._node_index.get(concept.strip().lower())
        if matching_node is None:
            return ""

        neighbors = list(self.graph.neighbors(matching_node))
        if not neighbors:
            return ""

        relations: list[str] = []
        for neighbor in neighbors[:5]:
            relation = _edge_relation(self.graph, matching_node, neighbor)
            relations.append(f"'{neighbor}' ({relation})")

        if relations:
            return f"Concept '{matching_node}' is linked to {', '.join(relations)}."

        return ""

    def get_context_for_concepts(self, concepts: list[str]) -> str:
        """Get graph context for multiple concepts in a single call.

        Deduplicates concepts (case-insensitive), processes up to 10, and
        joins individual context strings with spaces.

        Args:
            concepts: List of concept labels to look up.

        Returns:
            A space-separated string of context summaries, or empty if none found.
        """
        if not self._graph_enabled or not concepts:
            return ""

        seen: set[str] = set()
        unique_concepts: list[str] = []
        for c in concepts:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_concepts.append(c)

        contexts: list[str] = []
        for concept in unique_concepts[:10]:
            ctx = self.get_context(concept)
            if ctx:
                contexts.append(ctx)

        return " ".join(contexts)

    def get_facts(self, concepts: list[str]) -> list[str]:
        """Extract explicit relationship facts for a list of concepts.

        Traverses graph edges for each concept (up to 10 concepts, 8 neighbors
        each) and returns human-readable "Fact: A relation B" strings.
        If ontology fact labels are loaded, only matching edge relations
        are included.

        Args:
            concepts: List of concept strings to look up.

        Returns:
            A list of fact strings, capped at 30.
        """
        if not self._graph_enabled or not concepts:
            return []

        seen_in: set[str] = set()
        unique: list[str] = []
        for c in concepts:
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

            for neighbor in list(self.graph.neighbors(node_match))[:8]:
                relation = _edge_relation(self.graph, node_match, neighbor)
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
        """Resolve a concept to its taxonomy category.

        Checks direct keyword matches first, then falls back to the
        canonical synonym map.

        Args:
            concept: The concept label to classify (case-insensitive).

        Returns:
            The category name, or None if no match found.
        """
        if not self._taxonomy_enabled or not concept:
            return None

        concept_lower = concept.lower()

        if concept_lower in self._keyword_to_category:
            return self._keyword_to_category[concept_lower]

        canonical_map = self.taxonomy.get("canonical_map", {})
        canonical = canonical_map.get(concept_lower)
        if canonical is not None and canonical.lower() in self._keyword_to_category:
            return self._keyword_to_category[canonical.lower()]

        return None

    def get_all_categories(self) -> list[str]:
        """Return all taxonomy category names.

        Returns:
            A list of category name strings, or empty if taxonomy is not loaded.
        """
        if not self._taxonomy_enabled:
            return []
        cats = self.taxonomy.get("categories", {})
        return list(cats.keys())

    def get_canonical_map(self) -> dict[str, str]:
        """Return the synonym-to-canonical-term mapping from the taxonomy.

        Returns:
            A dictionary mapping lowercase synonyms to their canonical term.
        """
        if not self._taxonomy_enabled:
            return {}
        return self.taxonomy.get("canonical_map", {})


__all__ = ["GraphService"]
