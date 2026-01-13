"""Runtime taxonomy loader utilities.

These helpers live in the brain because taxonomy is used at runtime for:
- prompt injection (intent detection)
- concept normalization (graph branch)

They must degrade gracefully if taxonomy is missing.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_taxonomy_path() -> Path | None:
    try:
        from contextrouter.modules.ingestion.rag import get_assets_paths, load_config

        cfg = load_config()
        paths = get_assets_paths(cfg)
        p = paths.get("taxonomy")
        return p if isinstance(p, Path) else None
    except Exception:
        return None


@lru_cache(maxsize=2)
def load_taxonomy(taxonomy_path: str | None = None) -> dict[str, object] | None:
    """Load taxonomy JSON (cached)."""
    path: Path | None
    if taxonomy_path:
        path = Path(taxonomy_path)
    else:
        path = _default_taxonomy_path()

    if path is None or not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("Failed to load taxonomy from %s: %s", path, e)
        return None


def get_taxonomy_top_level_categories(*, taxonomy_path: str | None = None, limit: int = 20) -> str:
    """Return a formatted string of top-level taxonomy categories for prompt injection."""
    taxonomy = load_taxonomy(taxonomy_path)
    if not taxonomy:
        return ""

    cats = taxonomy.get("categories", {})
    if not isinstance(cats, dict) or not cats:
        return ""

    names = list(cats.keys())
    # Keep stable order but cap size
    names = names[: max(0, int(limit))]
    pretty = [n.replace("_", " ").title() for n in names]
    return "Taxonomy Categories: " + ", ".join(pretty)


def get_taxonomy_canonical_map(*, taxonomy_path: str | None = None) -> dict[str, str]:
    """Return synonym -> canonical mapping (lowercased keys)."""
    taxonomy = load_taxonomy(taxonomy_path)
    if not taxonomy:
        return {}
    cmap = taxonomy.get("canonical_map", {})
    if not isinstance(cmap, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in cmap.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        kk = k.strip().lower()
        vv = v.strip()
        if kk and vv:
            out[kk] = vv
    return out
