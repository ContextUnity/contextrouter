"""
Matcher pipeline stages.
Free deterministic stages that run before the paid RLM stage.
"""

from __future__ import annotations

from typing import Any

from contextcore import get_context_unit_logger

from .types import ProductMatch

logger = get_context_unit_logger(__name__)


def run_exact_match_stage(
    supplier_products: list[dict[str, Any]],
    site_products: list[dict[str, Any]],
) -> tuple[list[ProductMatch], list[dict[str, Any]]]:
    """
    Stage 1: Exact Match (EAN / Supplier SKU == Site SKU).

    Returns:
        matches: List of ProductMatch
        unmatched_suppliers: List of supplier products that still need matching
    """
    matches = []
    unmatched = []

    # Build exact matching indices
    site_skus = {
        str(p.get("manufacturer_sku", "")).strip().lower(): p
        for p in site_products
        if p.get("manufacturer_sku")
    }
    site_eans = {str(p.get("upc", "")).strip().lower(): p for p in site_products if p.get("upc")}

    for supplier in supplier_products:
        sku = str(supplier.get("sku", "")).strip().lower()
        ean = str(supplier.get("ean", "")).strip().lower()

        matched_site = None
        match_reason = ""

        # Priority 1: EAN/UPC Match
        if ean and ean in site_eans:
            matched_site = site_eans[ean]
            match_reason = "exact_ean"
        # Priority 2: Supplier SKU == manufacturer_sku
        elif sku and sku in site_skus:
            matched_site = site_skus[sku]
            match_reason = "exact_sku"

        if matched_site:
            matches.append(
                ProductMatch(
                    supplier_id=str(supplier["id"]),
                    supplier_name=supplier.get("name", ""),
                    site_id=str(matched_site["id"]),
                    site_name=matched_site.get("title", ""),
                    confidence=1.0,
                    match_type=match_reason,
                    notes=f"Auto-matched in Stage 1 ({match_reason})",
                )
            )
        else:
            unmatched.append(supplier)

    logger.info("Stage 1 (Exact Match): found %d matches", len(matches))
    return matches, unmatched


def run_normalized_match_stage(
    supplier_products: list[dict[str, Any]],
    site_products: list[dict[str, Any]],
) -> tuple[list[ProductMatch], list[dict[str, Any]]]:
    """
    Stage 2: Normalized Match (Gardener fields).

    If product_type, model_name, manufacturer_sku (optional),
    normalized_color, and normalized_size all match perfectly.

    Returns:
        matches: List of ProductMatch
        unmatched_suppliers: List of supplier products that still need matching
    """
    matches = []
    unmatched = []

    # Build signature index for site products
    site_signatures: dict[str, dict] = {}
    site_signature_counts: dict[str, int] = {}

    for site in site_products:
        sig = _build_normalized_signature(site)
        if sig:
            # We must handle duplicates. If two site products have the exact same
            # variant signature, it's ambiguous.
            site_signatures[sig] = site
            site_signature_counts[sig] = site_signature_counts.get(sig, 0) + 1

    for supplier in supplier_products:
        sig = _build_normalized_signature(supplier)

        if sig and sig in site_signatures and site_signature_counts[sig] == 1:
            matched_site = site_signatures[sig]
            matches.append(
                ProductMatch(
                    supplier_id=str(supplier["id"]),
                    supplier_name=supplier.get("name", ""),
                    site_id=str(matched_site["id"]),
                    site_name=matched_site.get("title", ""),
                    confidence=0.95,
                    match_type="normalized",
                    notes="Auto-matched in Stage 2 (Gardener Normalized)",
                )
            )
        else:
            if sig and site_signature_counts.get(sig, 0) > 1:
                logger.debug("Stage 2: ambiguous signature %s", sig)
            unmatched.append(supplier)

    logger.info("Stage 2 (Normalized Match): found %d matches", len(matches))
    return matches, unmatched


def _build_normalized_signature(product: dict[str, Any]) -> str | None:
    """
    Build a predictable signature string from normalized fields.
    Returns None if the product lacks sufficient normalized identity.
    """
    brand = str(product.get("brand", "")).strip().lower()
    product_type = str(product.get("product_type", "")).strip().lower()
    model_name = str(product.get("model_name", "")).strip().lower()

    # We NEED brand, product_type, and model_name to have an identity.
    if not brand or not product_type or not model_name:
        return None

    # Variables that distinguish variants
    # Wait, some items might not have size or color.
    # We still build the signature but it'll just be empty parts.
    color = str(product.get("normalized_color", "")).strip().lower()
    size = str(product.get("normalized_size", "")).strip().lower()
    gender = str(product.get("gender", "")).strip().lower()

    return f"{brand}::{product_type}::{model_name}::{gender}::{color}::{size}"
