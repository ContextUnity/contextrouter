"""
Matcher subgraph package.

Matcher: Product deduplication and matching.

Provides two matching modes:
- Regular: Small batch incremental matching
- RLM Bulk: Large-scale matching (50k+ products) using Recursive Language Models
"""

from .graph import (
    MatcherState,
    RLMBulkMatcherState,
    create_matcher_subgraph,
    create_rlm_bulk_matcher_subgraph,
)
from .nodes import MatchingNode
from .rlm_bulk import BulkMatchResult, ProductMatch, RLMBulkMatcher, rlm_bulk_match_node

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
]
