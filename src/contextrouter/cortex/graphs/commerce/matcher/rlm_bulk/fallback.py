"""Fallback chunked matching when RLM is not available."""

from __future__ import annotations

from typing import Any

from contextcore import get_context_unit_logger

from .types import BulkMatchResult, ProductMatch

logger = get_context_unit_logger(__name__)


async def fallback_chunked_match(
    supplier_products: list[dict[str, Any]],
    site_products: list[dict[str, Any]],
    confidence_threshold: float,
) -> BulkMatchResult:
    """Fallback matching when RLM is not available.

    Uses traditional chunked LLM calls instead of RLM recursion.
    Actually, just does simple sku-matching for now.
    """
    logger.info("Using fallback chunked matching (RLM not available)")

    # Simple SKU-based matching as fallback
    sku_index = {}
    for site in site_products:
        sku = site.get("sku", "").lower().strip()
        if sku:
            sku_index[sku] = site

    matches = []
    for supplier in supplier_products:
        supplier_sku = supplier.get("sku", "").lower().strip()
        supplier_id = str(supplier.get("id", supplier_sku))
        supplier_name = supplier.get("name", "")

        if supplier_sku in sku_index:
            site = sku_index[supplier_sku]
            matches.append(
                ProductMatch(
                    supplier_id=supplier_id,
                    supplier_name=supplier_name,
                    site_id=str(site.get("id", site.get("sku", ""))),
                    site_name=site.get("name", ""),
                    confidence=1.0,
                    match_type="sku_exact",
                    factors_matched=["sku"],
                    notes="Fallback SKU match",
                )
            )
        else:
            matches.append(
                ProductMatch(
                    supplier_id=supplier_id,
                    supplier_name=supplier_name,
                    site_id=None,
                    site_name=None,
                    confidence=0.0,
                    match_type="unmatched",
                    notes="No SKU match (RLM fallback mode)",
                )
            )

    return BulkMatchResult(
        total_supplier=len(supplier_products),
        total_site=len(site_products),
        matches=matches,
        unmatched=[
            p
            for p in supplier_products
            if not any(
                m.supplier_id == str(p.get("id", p.get("sku", ""))) and m.site_id for m in matches
            )
        ],
        stats={
            "mode": "fallback_sku_only",
            "match_rate": len([m for m in matches if m.site_id]) / len(supplier_products)
            if supplier_products
            else 0,
        },
    )
