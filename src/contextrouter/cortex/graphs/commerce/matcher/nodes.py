"""
Matcher node implementations.

Core matching logic for product deduplication and linking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextcore import BrainClient, ContextUnit

logger = logging.getLogger(__name__)


class MatchingNode:
    """
    Core matching logic.
    1. Pulls candidates from ContextBrain.
    2. Applies taxonomy filters.
    3. Uses LLM to decide on parity.
    """

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        site_product = state.get("product", {})
        query = f"Matching candidates for: {site_product.get('title')} {site_product.get('brand')}"

        brain = BrainClient()
        candidates = await brain.query_memory(ContextUnit(payload={"content": query}))

        matches = []
        for cand in candidates:
            if not self._taxonomy_match(site_product, cand.payload):
                continue

            if self._attribute_parity(site_product, cand.payload):
                matches.append(cand.payload)

        return {"matches": matches, "match_status": "linked" if matches else "no_parity"}

    def _taxonomy_match(self, site: Dict, supplier: Dict) -> bool:
        """Verify Category and Brand parity."""
        s_meta = site.get("metadata", {})
        p_meta = supplier.get("metadata", {})
        return s_meta.get("category") == p_meta.get("category") and site.get(
            "brand"
        ) == supplier.get("brand")

    def _attribute_parity(self, site: Dict, supplier: Dict) -> bool:
        """
        Check if specific attributes (Size, Color) match.
        Normalization happens in Brain, here we just check equality.
        """
        s_meta = site.get("metadata", {})
        p_meta = supplier.get("metadata", {})

        if not s_meta.get("size"):
            return True

        return s_meta.get("size") == p_meta.get("size")


async def fetch_unmatched_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch unmatched products from DB."""
    logger.info("Fetching unmatched products")
    # TODO: Query DB for unmatched products
    return {"products": []}


async def match_products_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Match products using MatchingNode."""
    matcher = MatchingNode()
    products = state.get("products", [])
    
    results = []
    for product in products:
        result = await matcher.process({"product": product})
        results.append(result)
    
    return {"match_results": results}


async def link_or_queue_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-link high confidence matches or add to review queue."""
    results = state.get("match_results", [])
    auto_linked = 0
    queued = 0
    
    for result in results:
        if result.get("match_status") == "linked":
            auto_linked += 1
        else:
            queued += 1
    
    logger.info(f"Matcher: {auto_linked} auto-linked, {queued} queued for review")
    return {"auto_linked": auto_linked, "queued": queued}
