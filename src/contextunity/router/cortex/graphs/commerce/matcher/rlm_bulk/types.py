"""Data types for RLM Bulk Matcher."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProductMatch:
    """Result of a single product match."""

    supplier_id: str
    supplier_name: str
    site_id: str | None
    site_name: str | None
    confidence: float
    match_type: str  # sku_exact, brand_model, name_fuzzy, semantic, unmatched
    factors_matched: list[str] = field(default_factory=list)
    factors_mismatched: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BulkMatchResult:
    """Result of bulk matching operation."""

    total_supplier: int
    total_site: int
    matches: list[ProductMatch]
    unmatched: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)
