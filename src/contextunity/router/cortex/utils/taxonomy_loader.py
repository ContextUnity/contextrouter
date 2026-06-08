"""Runtime taxonomy loader utilities.

These helpers live in the brain because taxonomy is used at runtime for:
- prompt injection (intent detection)
- concept normalization (graph branch)

They must degrade gracefully if taxonomy is missing.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.types import JsonDict, is_json_dict, is_object_dict

logger = get_contextunit_logger(__name__)


@runtime_checkable
class _TaxonomyAssetsModule(Protocol):
    """Dynamic module contract for taxonomy asset resolution."""

    def load_config(self) -> object:
        """Return the runtime ingestion config object."""
        ...

    def get_assets_paths(self, config: object) -> object:
        """Return a mapping of asset names to filesystem paths."""
        ...


def _import_taxonomy_assets_module() -> _TaxonomyAssetsModule:
    """Import the module that exposes taxonomy asset helpers."""
    try:
        module = importlib.import_module("contextunity.brain.ingestion.rag")
    except ImportError:
        module = importlib.import_module("contextunity.router.modules.ingestion.rag")
    if isinstance(module, _TaxonomyAssetsModule):
        return module
    raise TypeError("Taxonomy assets module does not expose the expected helper functions")


def _default_taxonomy_path() -> Path | None:
    """Resolve the default filesystem path to the taxonomy JSON asset.

    Dynamically imports the ingestion RAG package and loads the default configured asset paths.

    Returns:
        The resolved Path to the taxonomy asset, or None if it cannot be determined.
    """
    try:
        rag = _import_taxonomy_assets_module()
        cfg = rag.load_config()
        paths = rag.get_assets_paths(cfg)
        if not is_object_dict(paths):
            return None
        p = paths.get("taxonomy")
        return p if isinstance(p, Path) else None
    except Exception:
        return None


@lru_cache(maxsize=2)
def load_taxonomy(taxonomy_path: str | None = None) -> JsonDict | None:
    """Load and parse the taxonomy JSON file, caching the result.

    Args:
        taxonomy_path: Optional explicit file path to load. If omitted, the default
            taxonomy asset path is resolved.

    Returns:
        The parsed taxonomy dictionary, or None if loading/parsing fails.
    """
    path: Path | None
    if taxonomy_path:
        path = Path(taxonomy_path)
    else:
        path = _default_taxonomy_path()

    if path is None or not path.exists():
        return None

    try:
        data = json_loads(path.read_text(encoding="utf-8"))
        return data if is_json_dict(data) else None
    except Exception as e:
        logger.warning("Failed to load taxonomy from %s: %s", path, e)
        return None


def get_taxonomy_top_level_categories(*, taxonomy_path: str | None = None, limit: int = 20) -> str:
    """Retrieve and format a list of top-level taxonomy categories for prompt injection.

    Args:
        taxonomy_path: Optional explicit file path to load the taxonomy from.
        limit: The maximum number of category names to include. Defaults to 20.

    Returns:
        A formatted, human-readable string listing the top-level categories, or an empty
        string if the taxonomy is empty or missing.
    """
    taxonomy = load_taxonomy(taxonomy_path)
    if not taxonomy:
        return ""

    cats = taxonomy.get("categories")
    if not is_object_dict(cats) or not cats:
        return ""

    names: list[str] = list(cats)
    # Keep stable order but cap size
    names = names[: max(0, int(limit))]
    pretty = [n.replace("_", " ").title() for n in names]
    return "Taxonomy Categories: " + ", ".join(pretty)


def get_taxonomy_canonical_map(*, taxonomy_path: str | None = None) -> dict[str, str]:
    """Build a mapping from lowercased synonyms to their canonical category names.

    Args:
        taxonomy_path: Optional explicit file path to load the taxonomy from.

    Returns:
        A dictionary mapping lowercased synonym strings to canonical string values.
    """
    taxonomy = load_taxonomy(taxonomy_path)
    if not taxonomy:
        return {}
    cmap = taxonomy.get("canonical_map")
    if not is_object_dict(cmap):
        return {}
    out: dict[str, str] = {}
    for k, v in cmap.items():
        if not isinstance(v, str):
            continue
        kk = k.strip().lower()
        vv = v.strip()
        if kk and vv:
            out[kk] = vv
    return out
