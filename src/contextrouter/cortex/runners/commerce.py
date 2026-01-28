"""Runner entrypoint for Commerce-specific LangGraph agents.

This module provides entry points for invoking Commerce agents:
- Gardener: Product taxonomy classification (hybrid: batch + agentic + manual)
- Matcher: Canonical product matching
- Lexicon: Term extraction and normalization

Architecture:
    See plans/router/gardener-architecture.md for the hybrid batch/agentic/manual approach.

Usage:
    from contextrouter.cortex.runners.commerce import (
        run_gardener_batch,
        run_gardener_agentic,
        queue_for_manual_review,
    )

    # Batch mode (default, fast)
    results = await run_gardener_batch(tenant_id="tenant_123", batch_size=50)

    # Agentic mode (smart, for complex cases)
    result = await run_gardener_agentic(product_id=12345)

    # Manual mode (UI trigger)
    await queue_for_manual_review(product_id=12345, reason="low_confidence")
"""

from __future__ import annotations

import logging
from typing import Any

from contextrouter.core import get_core_config

logger = logging.getLogger(__name__)

# Confidence threshold for fallback to agentic mode
CONFIDENCE_THRESHOLD = 0.8


async def run_gardener_batch(
    tenant_id: str,
    batch_size: int = 50,
    trace_id: str | None = None,
    *,
    db_url: str | None = None,
    brain_url: str | None = None,
    fallback_to_agentic: bool = True,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Run Gardener agent on a batch of products (deterministic mode).

    Args:
        tenant_id: Tenant identifier.
        batch_size: Number of products to process.
        trace_id: Trace ID for observability.
        db_url: Override database URL (deprecated, use brain_url).
        brain_url: Brain gRPC URL.
        fallback_to_agentic: If True, queue low-confidence products for agentic processing.
        confidence_threshold: Minimum confidence to accept (default 0.7).

    Returns:
        Dict with processing results including fallback stats.
    """
    from ..graphs.commerce.gardener import compile_gardener_graph
    from ..graphs.commerce.queue import get_enrichment_queue

    config = get_core_config()

    # Initialize queue
    queue = get_enrichment_queue(config.redis.url)

    # Dequeue products
    product_ids = await queue.dequeue(
        tenant_id=tenant_id,
        batch_size=batch_size,
        batch_id=trace_id,
    )

    if not product_ids:
        logger.info("No products in enrichment queue")
        return {"status": "empty", "processed": 0}

    # Build input state
    input_state = {
        "tenant_id": tenant_id,
        "batch_size": batch_size,
        "trace_id": trace_id or f"gardener-{tenant_id}",
        "db_url": db_url or config.database.url,
        "brain_url": brain_url or config.brain.url,
        "products": [],
        "product_ids": product_ids,
    }

    # Compile and invoke graph (deterministic)
    graph = compile_gardener_graph()
    result = await graph.ainvoke(input_state)

    # Check confidence and fallback to agentic if needed
    fallback_count = 0
    if fallback_to_agentic:
        for product in result.get("products", []):
            confidence = product.get("enrichment", {}).get("taxonomy_confidence", 1.0)
            if confidence < confidence_threshold:
                product_id = product.get("id")
                await _queue_for_agentic(
                    product_id,
                    tenant_id=tenant_id,
                    reason=f"low_confidence:{confidence:.2f}",
                    trace_id=trace_id,
                )
                fallback_count += 1
                logger.info(
                    f"Product {product_id} queued for agentic (confidence={confidence:.2f})"
                )

    # Complete batch
    batch_state = await queue.complete_batch(trace_id) if trace_id else None

    return {
        "status": "completed",
        "processed": len(result.get("products", [])),
        "fallback_to_agentic": fallback_count,
        "errors": result.get("errors", []),
        "batch_state": batch_state,
    }


async def run_gardener_agentic(
    product_id: int,
    tenant_id: str | None = None,
    trace_id: str | None = None,
    *,
    brain_url: str | None = None,
    max_iterations: int = 10,
    fallback_to_manual: bool = True,
) -> dict[str, Any]:
    """Run Gardener in agentic mode for a single product (ReAct pattern).

    Use for:
    - Low confidence products from batch mode
    - Complex products requiring multi-step reasoning
    - Manual trigger from UI

    Args:
        product_id: Product to enrich.
        tenant_id: Tenant identifier.
        trace_id: Trace ID.
        brain_url: Brain gRPC URL.
        max_iterations: Safety limit for ReAct loops.
        fallback_to_manual: If True, queue for manual review if agent requests it.

    Returns:
        Dict with enrichment result.
    """
    from contextcore import BrainClient
    from langchain_core.messages import HumanMessage

    from ..graphs.commerce.gardener.agentic_graph import create_agentic_gardener_graph

    config = get_core_config()
    tenant_id = tenant_id or "default"

    # Fetch product from Brain
    brain = BrainClient(host=brain_url or config.brain.url)
    products = await brain.get_products(tenant_id, [product_id])

    if not products:
        return {"status": "error", "error": f"Product {product_id} not found"}

    product = products[0]

    # Build input state
    input_state = {
        "messages": [HumanMessage(content=f"Enrich this product: {product['name']}")],
        "product": product,
        "enrichment": product.get("enrichment", {}),
        "tenant_id": tenant_id,
        "trace_id": trace_id or f"agentic-{product_id}",
        "iteration": 0,
        "final_result": None,
    }

    # Run agentic graph
    graph = create_agentic_gardener_graph()
    result = await graph.ainvoke(input_state)

    # Check if agent requested human review
    last_msg = result.get("messages", [])[-1] if result.get("messages") else None
    needs_review = False
    review_reason = None

    if last_msg and hasattr(last_msg, "tool_calls"):
        for call in last_msg.tool_calls:
            if call.get("name") == "request_human_review":
                needs_review = True
                review_reason = call.get("args", {}).get("reason", "unknown")
                break

    if needs_review and fallback_to_manual:
        await queue_for_manual_review(
            product_id,
            tenant_id=tenant_id,
            reason=review_reason,
            trace_id=trace_id,
        )
        return {
            "status": "queued_for_review",
            "product_id": product_id,
            "reason": review_reason,
        }

    return {
        "status": "completed",
        "product_id": product_id,
        "enrichment": result.get("enrichment", {}),
        "iterations": result.get("iteration", 0),
    }


async def queue_for_manual_review(
    product_id: int,
    *,
    tenant_id: str = "default",
    reason: str = "manual",
    priority: str = "medium",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Queue product for manual review in UI.

    Args:
        product_id: Product to review.
        tenant_id: Tenant identifier.
        reason: Why manual review is needed.
        priority: Review priority (low, medium, high).
        trace_id: Trace ID.

    Returns:
        Dict with queue status.
    """
    from ..graphs.commerce.queue import get_enrichment_queue

    config = get_core_config()
    queue = get_enrichment_queue(config.redis.url)

    # Add to manual review queue with metadata
    await queue.add_to_manual_queue(
        product_id=product_id,
        tenant_id=tenant_id,
        metadata={
            "reason": reason,
            "priority": priority,
            "trace_id": trace_id,
        },
    )

    logger.info(f"Product {product_id} queued for manual review: {reason}")

    return {
        "status": "queued",
        "product_id": product_id,
        "queue": "manual_review",
        "reason": reason,
    }


async def _queue_for_agentic(
    product_id: int,
    tenant_id: str,
    reason: str,
    trace_id: str | None = None,
) -> None:
    """Internal: Queue product for agentic processing."""
    from ..graphs.commerce.queue import get_enrichment_queue

    config = get_core_config()
    queue = get_enrichment_queue(config.redis.url)

    await queue.add_to_agentic_queue(
        product_id=product_id,
        tenant_id=tenant_id,
        metadata={"reason": reason, "trace_id": trace_id},
    )


async def run_matcher_batch(
    tenant_id: str,
    product_ids: list[int],
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Run Matcher agent to find canonical products.

    Args:
        tenant_id: Tenant identifier.
        product_ids: Products to match.
        trace_id: Trace ID.

    Returns:
        Dict with matching results.
    """
    from ..graphs.commerce.matcher import compile_matcher_graph

    config = get_core_config()

    input_state = {
        "tenant_id": tenant_id,
        "product_ids": product_ids,
        "trace_id": trace_id or f"matcher-{tenant_id}",
        "brain_url": config.brain.url,
    }

    graph = compile_matcher_graph()
    result = await graph.ainvoke(input_state)

    return {
        "status": "completed",
        "matched": len(result.get("matched_products", [])),
        "unmatched": len(result.get("unmatched_products", [])),
    }


async def run_lexicon_extraction(
    tenant_id: str,
    text: str,
    domain: str = "commerce",
) -> dict[str, Any]:
    """Run Lexicon agent for term extraction.

    Args:
        tenant_id: Tenant identifier.
        text: Text to extract terms from.
        domain: Term domain (e.g., 'commerce', 'fashion').

    Returns:
        Dict with extracted terms.
    """
    from ..graphs.commerce.lexicon import compile_lexicon_graph

    input_state = {
        "tenant_id": tenant_id,
        "text": text,
        "domain": domain,
    }

    graph = compile_lexicon_graph()
    result = await graph.ainvoke(input_state)

    return {
        "terms": result.get("terms", []),
        "entities": result.get("entities", []),
    }


__all__ = [
    # Gardener (hybrid)
    "run_gardener_batch",
    "run_gardener_agentic",
    "queue_for_manual_review",
    # Other agents
    "run_matcher_batch",
    "run_lexicon_extraction",
    # Config
    "CONFIDENCE_THRESHOLD",
]
