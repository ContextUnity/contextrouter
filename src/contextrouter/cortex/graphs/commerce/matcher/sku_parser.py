"""SKU parsing utilities for extracting size, color, and other attributes.

This module provides pure utility functions to extract structured attributes
from supplier SKU/article codes. All domain-specific data (colors, sizes)
must be provided as parameters — loaded from taxonomy at runtime.

NO HARDCODED DOMAIN VALUES. Use data_loaders.py to load from taxonomy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SkuAttributes",
    "SkuParser",
    "parse_sku_attributes",
    "normalize_sku",
]


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class SkuAttributes:
    """Attributes extracted from SKU/article code."""

    sku_size: str | None = None
    sku_color: str | None = None
    sku_normalized: str = ""
    brand_prefix: str | None = None
    model_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sku_size": self.sku_size,
            "sku_color": self.sku_color,
            "sku_normalized": self.sku_normalized,
            "brand_prefix": self.brand_prefix,
            "model_code": self.model_code,
        }


# ============================================================================
# SKU Parser
# ============================================================================


class SkuParser:
    """SKU parser with taxonomy-based color and size extraction.

    Color and size mappings must be loaded from taxonomy and passed
    to the constructor. No hardcoded domain values.

    Example:
        # Load mappings from taxonomy
        color_mappings = await load_colors_from_taxonomy()
        size_values = await load_sizes_from_taxonomy()

        parser = SkuParser(
            color_mappings=color_mappings,
            size_values=size_values,
        )
        attrs = parser.parse("PROD-015T-black-M")
    """

    def __init__(
        self,
        color_mappings: dict[str, str] | None = None,
        size_values: set[str] | None = None,
    ):
        """Initialize parser with taxonomy data.

        Args:
            color_mappings: Dict mapping color variants to normalized colors.
                            Example: {"blk": "black", "чорний": "black"}
            size_values: Set of valid size values.
                         Example: {"XS", "S", "M", "L", "XL", "42", "44"}
        """
        self._color_mappings = color_mappings or {}
        self._size_values = size_values or set()

        # Build color pattern from mappings
        if self._color_mappings:
            color_patterns = "|".join(re.escape(abbr) for abbr in self._color_mappings.keys())
            self._color_pattern = re.compile(rf"\b({color_patterns})\b", re.IGNORECASE)
        else:
            self._color_pattern = None

        # Build size pattern from values
        if self._size_values:
            # Sort by length descending to match longer patterns first
            sorted_sizes = sorted(self._size_values, key=len, reverse=True)
            size_patterns = "|".join(re.escape(s) for s in sorted_sizes)
            self._size_pattern = re.compile(rf"\b({size_patterns})\b", re.IGNORECASE)
        else:
            self._size_pattern = None

    def parse(self, sku: str) -> SkuAttributes:
        """Extract attributes from SKU.

        Args:
            sku: Raw SKU/article code.

        Returns:
            SkuAttributes with extracted values.
        """
        if not sku:
            return SkuAttributes()

        result = SkuAttributes(sku_normalized=normalize_sku(sku))

        # Extract color using taxonomy mappings
        if self._color_pattern:
            match = self._color_pattern.search(sku)
            if match:
                found = match.group(1).lower()
                result.sku_color = self._color_mappings.get(found, found)

        # Extract size using taxonomy values
        if self._size_pattern:
            match = self._size_pattern.search(sku)
            if match:
                result.sku_size = match.group(1).upper()

        # Extract brand prefix (first segment)
        segments = re.split(r"[-_\s]+", sku)
        if len(segments) >= 2:
            result.brand_prefix = segments[0]
            if len(segments) >= 3:
                result.model_code = "-".join(segments[1:3])

        return result


# ============================================================================
# Standalone Functions
# ============================================================================


def parse_sku_attributes(
    sku: str,
    color_mappings: dict[str, str] | None = None,
    size_values: set[str] | None = None,
) -> SkuAttributes:
    """Extract attributes from SKU (convenience function).

    For repeated parsing, use SkuParser class directly for better performance.

    Args:
        sku: Raw SKU/article code.
        color_mappings: Dict mapping color variants to normalized colors.
        size_values: Set of valid size values.

    Returns:
        SkuAttributes with extracted values.
    """
    parser = SkuParser(color_mappings=color_mappings, size_values=size_values)
    return parser.parse(sku)


def normalize_sku(sku: str) -> str:
    """Normalize SKU for comparison: lowercase, alphanumeric only.

    Args:
        sku: Raw SKU string.

    Returns:
        Normalized SKU with only lowercase alphanumeric characters.

    Examples:
        >>> normalize_sku("PROD-015T-black-XL")
        'prod015tblackxl'
    """
    return re.sub(r"[^a-z0-9]", "", sku.lower())
