"""
Commerce state definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class CommerceState(TypedDict):
    """State for Commerce graph.

    The Commerce graph routes to subgraphs based on intent:
    - enrich → Gardener subgraph
    - generate_content → Lexicon subgraph
    - match_products → Matcher subgraph
    """

    # Routing
    intent: str  # enrich, generate_content, match_products, search

    # Common config
    tenant_id: str
    db_url: str
    batch_size: int
    prompts_dir: str

    # Input
    product_ids: List[int]

    # Trace
    trace_id: str

    # Output (filled by subgraph)
    result: Dict[str, Any]
    errors: List[str]
