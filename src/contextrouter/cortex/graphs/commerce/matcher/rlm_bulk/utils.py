"""Utility functions for RLM Bulk Matcher."""

from __future__ import annotations

# Fields to keep for matching — everything else is noise (images, urls, description...)
# Supplier products use: name, sku, brand
# Site (Oscar) products use: title, upc, parent_title, manufacturer_sku
_KEEP_FIELDS = frozenset(
    {
        "id",
        "name",
        "title",
        "parent_title",
        "brand",
        "sku",
        "upc",
        "ean",
        "manufacturer_sku",
        "category",
        "normalized_category",
        "price_retail_uah",
        "product_type",
        "structure",
    }
)
_KEEP_PARAMS = frozenset({"color", "size"})


def slim_products(products: list[dict]) -> list[dict]:
    """Strip heavy fields (images, urls, description) to reduce token usage."""
    slim = []
    for p in products:
        item = {k: v for k, v in p.items() if k in _KEEP_FIELDS or k == "params"}
        if "params" in item and isinstance(item["params"], dict):
            item["params"] = {k: v for k, v in item["params"].items() if k in _KEEP_PARAMS and v}
            if not item["params"]:
                del item["params"]
        slim.append(item)
    return slim
