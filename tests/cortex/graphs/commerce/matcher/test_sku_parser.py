"""Tests for SKU parser utilities."""

import pytest

from contextrouter.cortex.graphs.commerce.matcher.sku_parser import (
    SkuAttributes,
    SkuParser,
    normalize_sku,
    parse_sku_attributes,
)


# Sample taxonomy data for tests (would be loaded from DB in production)
SAMPLE_COLOR_MAPPINGS = {
    "black": "black",
    "blk": "black",
    "white": "white",
    "grey": "gray",
    "gray": "gray",
    "orange": "orange",
    "khaki": "khaki",
    "yellow": "yellow",
    "navy": "navy",
    "red": "red",
}

SAMPLE_SIZE_VALUES = {"XS", "S", "M", "L", "XL", "XXL", "42", "44", "46"}


class TestSkuParser:
    """Tests for SkuParser class."""

    def test_parser_with_color_mappings(self):
        """Test parser extracts colors using provided mappings."""
        parser = SkuParser(color_mappings=SAMPLE_COLOR_MAPPINGS)
        attrs = parser.parse("PROD-123-orange")
        assert attrs.sku_color == "orange"

    def test_parser_normalizes_color_variants(self):
        """Test that color variants are normalized via mappings."""
        parser = SkuParser(color_mappings=SAMPLE_COLOR_MAPPINGS)
        attrs = parser.parse("PROD-108-grey")
        assert attrs.sku_color == "gray"

    def test_parser_with_size_values(self):
        """Test parser extracts sizes using provided values."""
        parser = SkuParser(size_values=SAMPLE_SIZE_VALUES)
        attrs = parser.parse("PROD-015T-M")
        assert attrs.sku_size == "M"

    def test_parser_with_both_mappings(self):
        """Test parser extracts both color and size."""
        parser = SkuParser(
            color_mappings=SAMPLE_COLOR_MAPPINGS,
            size_values=SAMPLE_SIZE_VALUES,
        )
        attrs = parser.parse("PROD-015T-black-M")
        assert attrs.sku_color == "black"
        assert attrs.sku_size == "M"

    def test_parser_no_mappings_returns_empty(self):
        """Test parser without mappings doesn't extract color/size."""
        parser = SkuParser()
        attrs = parser.parse("PROD-015T-black-M")
        assert attrs.sku_color is None
        assert attrs.sku_size is None
        # But normalized SKU is still extracted
        assert attrs.sku_normalized == "prod015tblackm"

    def test_parser_extracts_brand_prefix(self):
        """Test brand prefix extraction."""
        parser = SkuParser()
        attrs = parser.parse("BRAND-015T-black-XL")
        assert attrs.brand_prefix == "BRAND"

    def test_parser_extracts_model_code(self):
        """Test model code extraction."""
        parser = SkuParser()
        attrs = parser.parse("BRAND-MODEL-001-black")
        assert attrs.model_code == "MODEL-001"

    def test_parser_empty_sku(self):
        """Test empty SKU returns empty result."""
        parser = SkuParser()
        attrs = parser.parse("")
        assert attrs == SkuAttributes()

    def test_parser_numeric_sku(self):
        """Test numeric-only SKU."""
        parser = SkuParser(size_values=SAMPLE_SIZE_VALUES)
        attrs = parser.parse("39602000")
        assert attrs.sku_color is None
        assert attrs.sku_size is None


class TestParseSkuAttributes:
    """Tests for parse_sku_attributes convenience function."""

    def test_with_mappings(self):
        """Test convenience function with mappings."""
        attrs = parse_sku_attributes(
            "PROD-black-M",
            color_mappings=SAMPLE_COLOR_MAPPINGS,
            size_values=SAMPLE_SIZE_VALUES,
        )
        assert attrs.sku_color == "black"
        assert attrs.sku_size == "M"

    def test_without_mappings(self):
        """Test convenience function without mappings."""
        attrs = parse_sku_attributes("PROD-black-M")
        assert attrs.sku_color is None
        assert attrs.sku_size is None
        assert attrs.sku_normalized == "prodblackm"


class TestNormalizeSku:
    """Tests for normalize_sku function."""

    def test_removes_dashes(self):
        """Test that dashes are removed."""
        assert normalize_sku("ABC-109-orange") == "abc109orange"

    def test_removes_underscores(self):
        """Test that underscores are removed."""
        assert normalize_sku("PRODUCT_CODE_123") == "productcode123"

    def test_lowercases(self):
        """Test that result is lowercase."""
        assert normalize_sku("UPPERCASE") == "uppercase"

    def test_keeps_alphanumeric(self):
        """Test that alphanumeric characters are kept."""
        assert normalize_sku("ABC-123") == "abc123"

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert normalize_sku("") == ""


class TestSkuAttributes:
    """Tests for SkuAttributes dataclass."""

    def test_as_dict(self):
        """Test as_dict method."""
        attrs = SkuAttributes(
            sku_size="M",
            sku_color="black",
            sku_normalized="prod015tblackm",
            brand_prefix="PROD",
            model_code="015T",
        )
        result = attrs.as_dict()
        assert result == {
            "sku_size": "M",
            "sku_color": "black",
            "sku_normalized": "prod015tblackm",
            "brand_prefix": "PROD",
            "model_code": "015T",
        }

    def test_default_values(self):
        """Test default values."""
        attrs = SkuAttributes()
        assert attrs.sku_size is None
        assert attrs.sku_color is None
        assert attrs.sku_normalized == ""
        assert attrs.brand_prefix is None
        assert attrs.model_code is None
