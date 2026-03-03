"""RLM Bulk Matcher - Deep product matching using Recursive Language Models."""

from .matcher import RLMBulkMatcher
from .node import rlm_bulk_match_node
from .prompts import DEEP_MATCHING_INSTRUCTIONS
from .types import BulkMatchResult, ProductMatch

__all__ = [
    "RLMBulkMatcher",
    "ProductMatch",
    "BulkMatchResult",
    "rlm_bulk_match_node",
    "DEEP_MATCHING_INSTRUCTIONS",
]
