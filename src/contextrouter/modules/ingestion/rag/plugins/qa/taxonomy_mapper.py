"""Taxonomy mapping component for QA plugin."""

from __future__ import annotations

import logging

from contextrouter.core.types import StructData

logger = logging.getLogger(__name__)


class TaxonomyMapper:
    """Handles mapping of content to taxonomy categories."""

    def __init__(self, taxonomy: StructData | None = None) -> None:
        self.taxonomy = taxonomy or {}

    def _get_taxonomy_categories(self, taxonomy: StructData) -> list[str]:
        """Extract category names from taxonomy."""
        categories: list[str] = []

        # Extract from different possible taxonomy formats
        if isinstance(taxonomy, dict):
            # Look for categories in various places
            for key in ["categories", "topics", "subjects"]:
                if key in taxonomy and isinstance(taxonomy[key], list):
                    categories.extend(str(cat) for cat in taxonomy[key] if cat)

        return categories

    def map_to_taxonomy(self, content: str, taxonomy: StructData | None = None) -> list[str]:
        """Map content to relevant taxonomy categories."""
        if not taxonomy and not self.taxonomy:
            return []

        taxonomy_data = taxonomy or self.taxonomy
        categories = self._get_taxonomy_categories(taxonomy_data)

        if not categories:
            return []

        # Simple keyword matching - can be enhanced with ML
        content_lower = content.lower()
        matched_categories = []

        for category in categories:
            category_lower = category.lower()
            # Check if category keywords appear in content
            if any(word in content_lower for word in category_lower.split()):
                matched_categories.append(category)

        return matched_categories

    def update_taxonomy(self, new_taxonomy: StructData) -> None:
        """Update the taxonomy data."""
        self.taxonomy = new_taxonomy
