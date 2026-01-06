"""Ingestion graph: config loading step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contextrouter.core import get_core_config
from contextrouter.modules.ingestion.rag import (
    RagIngestionConfig,
    ensure_directories_exist,
    get_assets_paths,
    load_config,
)


def load_ingestion_config(state: dict[str, Any]) -> dict[str, Any]:
    """Load RagIngestionConfig and resolve assets paths.

    Expected state inputs:
    - ingestion_config_path: str | None
    - only_types: list[str] | None
    - overwrite: bool
    - workers: int
    """

    core_cfg = get_core_config()

    raw_path = state.get("ingestion_config_path")
    cfg_path: Path | None = None
    if isinstance(raw_path, str) and raw_path.strip():
        cfg_path = Path(raw_path.strip())

    cfg: RagIngestionConfig = load_config(cfg_path)
    paths = get_assets_paths(cfg)
    ensure_directories_exist(paths)

    only_types = state.get("only_types")
    if not isinstance(only_types, list) or not only_types:
        only_types = ["video", "book", "qa", "web", "knowledge"]
    only_types = [t for t in only_types if isinstance(t, str) and t.strip()]

    return {
        "core_cfg": core_cfg,
        "ingestion_cfg": cfg,
        "assets_paths": {k: str(v) for k, v in paths.items()},
        "only_types": only_types,
    }


__all__ = ["load_ingestion_config"]
