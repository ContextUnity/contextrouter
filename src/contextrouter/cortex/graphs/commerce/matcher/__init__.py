"""
Matcher subgraph package.

Matcher: Product deduplication and matching.
"""

from .graph import MatcherState, create_matcher_subgraph
from .nodes import MatchingNode

__all__ = ["create_matcher_subgraph", "MatcherState", "MatchingNode"]
