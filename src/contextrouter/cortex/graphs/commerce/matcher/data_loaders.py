"""Taxonomy and Knowledge Graph loaders for RLM matching.

This module provides helpers to load taxonomy domains (category, color, size, gender)
and Knowledge Graph edges from ContextBrain for use in RLM product matching.

The loaded data can be injected as variables in the RLM REPL environment for
O(1) lookups during matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "TaxonomyData",
    "KnowledgeGraphData",
    "RLMDataLoader",
    "load_taxonomy_for_rlm",
    "load_knowledge_graph_for_rlm",
]


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class TaxonomyNode:
    """Single taxonomy node (category, color, size, etc.)."""

    path: str
    label: str
    keywords: list[str] = field(default_factory=list)
    parent_path: str | None = None


@dataclass
class TaxonomyData:
    """Taxonomy data for RLM matching.

    Contains indexed lookups for efficient matching:
    - categories: category path → TaxonomyNode
    - colors: color slug → normalized color
    - sizes: size value → normalized size
    - genders: gender value → normalized gender
    - product_types: product type → canonical form
    """

    categories: dict[str, TaxonomyNode] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    sizes: dict[str, str] = field(default_factory=dict)
    genders: dict[str, str] = field(default_factory=dict)
    product_types: dict[str, str] = field(default_factory=dict)

    def as_rlm_variables(self) -> dict[str, Any]:
        """Convert to dict suitable for RLM variable injection.

        Returns:
            Dict with indexed lookups for use in RLM REPL.
        """
        return {
            "taxonomy_categories": {
                path: {"label": node.label, "keywords": node.keywords}
                for path, node in self.categories.items()
            },
            "taxonomy_colors": self.colors,
            "taxonomy_sizes": self.sizes,
            "taxonomy_genders": self.genders,
            "taxonomy_product_types": self.product_types,
        }


@dataclass
class KnowledgeGraphData:
    """Knowledge Graph data for RLM matching.

    Contains indexed lookups for matching:
    - brand_products: brand_id → list of product_ids
    - color_products: color_slug → list of product_ids
    - gender_products: gender → list of product_ids
    - technology_products: technology_slug → list of product_ids
    """

    brand_products: dict[str, list[str]] = field(default_factory=dict)
    color_products: dict[str, list[str]] = field(default_factory=dict)
    gender_products: dict[str, list[str]] = field(default_factory=dict)
    technology_products: dict[str, list[str]] = field(default_factory=dict)
    # Direct product → attributes mapping
    product_attributes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_rlm_variables(self) -> dict[str, Any]:
        """Convert to dict suitable for RLM variable injection."""
        return {
            "kg_brand_products": self.brand_products,
            "kg_color_products": self.color_products,
            "kg_gender_products": self.gender_products,
            "kg_technology_products": self.technology_products,
            "kg_product_attributes": self.product_attributes,
        }


# ============================================================================
# Data Loaders
# ============================================================================


class RLMDataLoader:
    """Loader for taxonomy and KG data for RLM matching.

    Connects to ContextBrain to fetch:
    - Taxonomy domains (category, color, size, gender)
    - Knowledge Graph edges (MADE_BY, HAS_COLOR, FOR_GENDER, USES)
    """

    def __init__(
        self,
        brain_url: str | None = None,
        tenant_id: str = "",
    ):
        """Initialize data loader.

        Args:
            brain_url: ContextBrain gRPC URL. Uses default if not specified.
            tenant_id: Tenant ID for filtering data. Must be provided by caller — no default.
        """
        if not tenant_id:
            raise ValueError("tenant_id must be provided to RLMDataLoader — it is project-specific")
        self._brain_url = brain_url or "localhost:50051"
        self._tenant_id = tenant_id

    async def load_taxonomy(self) -> TaxonomyData:
        """Load taxonomy data from ContextBrain.

        Returns:
            TaxonomyData with indexed lookups for all domains.
        """
        try:
            from contextcore import BrainClient
        except ImportError:
            logger.warning("contextcore not installed, returning empty taxonomy")
            return TaxonomyData()

        from contextrouter.core.brain_token import get_brain_service_token

        data = TaxonomyData()

        try:
            async with BrainClient(self._brain_url, token=get_brain_service_token()) as client:
                # Load categories (both singular and plural forms)
                for domain in ["category", "categories"]:
                    items = await client.list_taxonomy(
                        tenant_id=self._tenant_id,
                        domain=domain,
                    )
                    for item in items:
                        path = item.get("path", "")
                        data.categories[path] = TaxonomyNode(
                            path=path,
                            label=item.get("label", path.split(".")[-1]),
                            keywords=item.get("keywords", []),
                            parent_path=".".join(path.split(".")[:-1]) or None,
                        )

                # Load colors
                for domain in ["color", "colors"]:
                    items = await client.list_taxonomy(
                        tenant_id=self._tenant_id,
                        domain=domain,
                    )
                    for item in items:
                        slug = item.get("path", "")
                        # Index by slug and all keywords
                        data.colors[slug] = slug
                        for kw in item.get("keywords", []):
                            data.colors[kw.lower()] = slug

                # Load sizes
                for domain in ["size", "sizes"]:
                    items = await client.list_taxonomy(
                        tenant_id=self._tenant_id,
                        domain=domain,
                    )
                    for item in items:
                        value = item.get("path", "")
                        data.sizes[value] = value
                        for kw in item.get("keywords", []):
                            data.sizes[kw.upper()] = value

                # Load genders
                for domain in ["gender", "genders"]:
                    items = await client.list_taxonomy(
                        tenant_id=self._tenant_id,
                        domain=domain,
                    )
                    for item in items:
                        value = item.get("path", "")
                        data.genders[value] = value
                        for kw in item.get("keywords", []):
                            data.genders[kw.lower()] = value

        except Exception as e:
            logger.error("Failed to load taxonomy from Brain: %s", e)

        logger.info(
            "Loaded taxonomy: %s categories, %s colors, %s sizes, %s genders",
            len(data.categories),
            len(data.colors),
            len(data.sizes),
            len(data.genders),
        )

        return data

    async def load_knowledge_graph(self) -> KnowledgeGraphData:
        """Load Knowledge Graph edges from ContextBrain.

        Loads edges for:
        - MADE_BY (product → brand)
        - HAS_COLOR (product → color)
        - FOR_GENDER (product → gender)
        - USES (product → technology)

        Returns:
            KnowledgeGraphData with indexed lookups.
        """
        try:
            from contextcore import BrainClient
        except ImportError:
            logger.warning("contextcore not installed, returning empty KG")
            return KnowledgeGraphData()

        from contextrouter.core.brain_token import get_brain_service_token

        data = KnowledgeGraphData()

        try:
            async with BrainClient(self._brain_url, token=get_brain_service_token()) as client:
                # Load brand edges
                edges = await client.query_edges(
                    tenant_id=self._tenant_id,
                    relation="MADE_BY",
                )
                for edge in edges:
                    brand_id = edge.get("target_id", "")
                    product_id = edge.get("source_id", "")
                    if brand_id not in data.brand_products:
                        data.brand_products[brand_id] = []
                    data.brand_products[brand_id].append(product_id)

                    # Also store in product_attributes
                    if product_id not in data.product_attributes:
                        data.product_attributes[product_id] = {}
                    data.product_attributes[product_id]["brand"] = brand_id

                # Load color edges
                edges = await client.query_edges(
                    tenant_id=self._tenant_id,
                    relation="HAS_COLOR",
                )
                for edge in edges:
                    color = edge.get("target_id", "").replace("color:", "")
                    product_id = edge.get("source_id", "")
                    if color not in data.color_products:
                        data.color_products[color] = []
                    data.color_products[color].append(product_id)

                    if product_id not in data.product_attributes:
                        data.product_attributes[product_id] = {}
                    data.product_attributes[product_id]["color"] = color

                # Load gender edges
                edges = await client.query_edges(
                    tenant_id=self._tenant_id,
                    relation="FOR_GENDER",
                )
                for edge in edges:
                    gender = edge.get("target_id", "").replace("gender:", "")
                    product_id = edge.get("source_id", "")
                    if gender not in data.gender_products:
                        data.gender_products[gender] = []
                    data.gender_products[gender].append(product_id)

                    if product_id not in data.product_attributes:
                        data.product_attributes[product_id] = {}
                    data.product_attributes[product_id]["gender"] = gender

                # Load technology edges
                edges = await client.query_edges(
                    tenant_id=self._tenant_id,
                    relation="USES",
                )
                for edge in edges:
                    tech = edge.get("target_id", "").replace("technology:", "")
                    product_id = edge.get("source_id", "")
                    if tech not in data.technology_products:
                        data.technology_products[tech] = []
                    data.technology_products[tech].append(product_id)

                    if product_id not in data.product_attributes:
                        data.product_attributes[product_id] = {}
                    if "technologies" not in data.product_attributes[product_id]:
                        data.product_attributes[product_id]["technologies"] = []
                    data.product_attributes[product_id]["technologies"].append(tech)

        except Exception as e:
            logger.error("Failed to load KG from Brain: %s", e)

        logger.info(
            "Loaded KG: %s brands, %s colors, %s genders, %s technologies, %s products with attributes",
            len(data.brand_products),
            len(data.color_products),
            len(data.gender_products),
            len(data.technology_products),
            len(data.product_attributes),
        )

        return data


# ============================================================================
# Convenience Functions
# ============================================================================


async def load_taxonomy_for_rlm(
    brain_url: str | None = None,
    tenant_id: str = "",
) -> dict[str, Any]:
    """Load taxonomy data formatted for RLM variable injection.

    Args:
        brain_url: ContextBrain gRPC URL.
        tenant_id: Tenant ID for filtering (required — project-specific, no default).

    Returns:
        Dict with taxonomy lookups suitable for RLM variables.
    """
    loader = RLMDataLoader(brain_url=brain_url, tenant_id=tenant_id)
    data = await loader.load_taxonomy()
    return data.as_rlm_variables()


async def load_knowledge_graph_for_rlm(
    brain_url: str | None = None,
    tenant_id: str = "",
) -> dict[str, Any]:
    """Load KG data formatted for RLM variable injection.

    Args:
        brain_url: ContextBrain gRPC URL.
        tenant_id: Tenant ID for filtering (required — project-specific, no default).

    Returns:
        Dict with KG lookups suitable for RLM variables.
    """
    loader = RLMDataLoader(brain_url=brain_url, tenant_id=tenant_id)
    data = await loader.load_knowledge_graph()
    return data.as_rlm_variables()
