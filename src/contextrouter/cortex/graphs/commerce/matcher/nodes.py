"""
Matcher node implementations.

Core matching logic for product deduplication and linking.
All DB operations happen in Commerce's process via BiDi stream tools.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def fetch_unmatched_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch unmatched products via BiDi tool (export_unmatched_products)."""
    logger.info("Fetching unmatched products")

    from contextrouter.service.stream_executors import get_stream_executor_manager

    tenant_id = state.get("tenant_id", "traverse")
    manager = get_stream_executor_manager()

    if not manager.is_available(tenant_id, "export_unmatched_products"):
        logger.warning("No BiDi stream for project '%s' — cannot fetch products", tenant_id)
        return {"products": []}

    try:
        result = await manager.execute(tenant_id, "export_unmatched_products", {}, timeout=120)
        products = result.get("products", [])
        logger.info("Fetched %d unmatched products for matching", len(products))
        return {"products": products}
    except Exception as e:
        logger.exception("Failed to fetch unmatched products: %s", e)
        return {"products": []}


async def match_products_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Match products — queue all unmatched for human review.

    This node marks all unmatched products as `pending_review`
    so they appear in the AI Matcher HITL panel.

    For AI-powered matching (RLM), use create_rlm_bulk_matcher_subgraph().
    """
    products = state.get("products", [])
    if not products:
        logger.info("No products to match")
        return {"match_results": []}

    # All unmatched products → pending_review for HITL
    results = []
    for p in products:
        results.append(
            {
                "id": p.get("id"),
                "name": p.get("name", ""),
                "brand": p.get("brand", ""),
                "sku": p.get("sku", ""),
                "ean": p.get("ean", ""),
                "match_status": "pending_review",
            }
        )

    logger.info("Prepared %d products for review", len(results))
    return {"match_results": results}


async def link_or_queue_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Update matching_status to pending_review via BiDi tool."""
    results = state.get("match_results", [])
    if not results:
        logger.info("No match results to process")
        return {"auto_linked": 0, "queued": 0}

    from contextrouter.service.stream_executors import get_stream_executor_manager

    tenant_id = state.get("tenant_id", "traverse")
    manager = get_stream_executor_manager()

    product_ids = [r["id"] for r in results if r.get("id")]

    if not product_ids:
        logger.warning("No product IDs in match results")
        return {"auto_linked": 0, "queued": 0}

    # Use bulk_link_products BiDi tool to update status
    if manager.is_available(tenant_id, "bulk_link_products"):
        try:
            await manager.execute(
                tenant_id,
                "bulk_link_products",
                {
                    "action": "set_pending_review",
                    "product_ids": product_ids,
                },
                timeout=120,
            )
            logger.info("Set %d products to pending_review", len(product_ids))
        except Exception as e:
            logger.exception("Failed to update product statuses: %s", e)
    else:
        logger.warning("bulk_link_products BiDi tool not available for '%s'", tenant_id)

    logger.info("Matcher: 0 auto-linked, %d queued for review", len(product_ids))
    return {"auto_linked": 0, "queued": len(product_ids)}
