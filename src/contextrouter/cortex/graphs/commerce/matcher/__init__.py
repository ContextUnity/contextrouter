"""
Matcher subgraph package.

Matcher: Product deduplication and matching.

Provides two matching modes:
- Regular: Small batch incremental matching
- RLM Bulk: Large-scale matching (50k+ products) using Recursive Language Models
"""

from .data_loaders import (
    KnowledgeGraphData,
    RLMDataLoader,
    TaxonomyData,
    load_knowledge_graph_for_rlm,
    load_taxonomy_for_rlm,
)
from .graph import (
    MatcherState,
    RLMBulkMatcherState,
    create_matcher_subgraph,
    create_rlm_bulk_matcher_subgraph,
)
from .nodes import MatchingNode
from .rlm_bulk import BulkMatchResult, ProductMatch, RLMBulkMatcher, rlm_bulk_match_node
from .sku_parser import (
    SkuAttributes,
    SkuParser,
    normalize_sku,
    parse_sku_attributes,
)

__all__ = [
    # Regular matcher
    "create_matcher_subgraph",
    "MatcherState",
    "MatchingNode",
    # RLM bulk matcher
    "create_rlm_bulk_matcher_subgraph",
    "RLMBulkMatcherState",
    "RLMBulkMatcher",
    "ProductMatch",
    "BulkMatchResult",
    "rlm_bulk_match_node",
    # SKU parsing utilities
    "SkuParser",
    "SkuAttributes",
    "parse_sku_attributes",
    "normalize_sku",
    # Data loaders for RLM
    "RLMDataLoader",
    "TaxonomyData",
    "KnowledgeGraphData",
    "load_taxonomy_for_rlm",
    "load_knowledge_graph_for_rlm",
]
