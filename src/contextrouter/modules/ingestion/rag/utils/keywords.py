"""Keyword taxonomy utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default relative to package root (packages/contextrouter/)
DEFAULT_KEYWORDS_PATH = Path("assets/taxonomy.json")


def load_keyword_taxonomy(path: Path | None = None) -> dict | None:
    taxonomy_path = path or DEFAULT_KEYWORDS_PATH
    if not taxonomy_path.exists():
        logger.debug("No keyword taxonomy found at %s", taxonomy_path)
        return None

    try:
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        logger.info("Loaded keyword taxonomy")
        return taxonomy
    except Exception as e:
        logger.warning("Failed to load keyword taxonomy: %s", e)
        return None


def get_taxonomy_keywords(taxonomy: dict | None) -> list[str]:
    if not taxonomy:
        return []

    if "all_keywords" in taxonomy and isinstance(taxonomy["all_keywords"], list):
        return taxonomy["all_keywords"]

    keywords: list[str] = []
    categories = taxonomy.get("categories", {})

    if isinstance(categories, dict):
        for cat_data in categories.values():
            if isinstance(cat_data, dict):
                kws = cat_data.get("keywords", [])
                if isinstance(kws, list):
                    keywords.extend([str(k) for k in kws])
    elif isinstance(categories, list):
        for cat_data in categories:
            if isinstance(cat_data, dict):
                kws = cat_data.get("keywords", [])
                if isinstance(kws, list):
                    keywords.extend([str(k) for k in kws])

    return keywords
