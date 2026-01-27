"""Transformer: CleanText -> taxonomy.json."""

from __future__ import annotations

import logging
from typing import Any

from contextrouter.core import BaseTransformer, Config, ContextUnit
from contextrouter.modules.ingestion.rag.config import get_assets_paths, load_config
from contextrouter.modules.ingestion.rag.core.utils import resolve_workers
from contextrouter.modules.ingestion.rag.processors.taxonomy_builder import (
    build_taxonomy,
)
from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig

logger = logging.getLogger(__name__)


def build_taxonomy_from_clean_text(
    *,
    config: RagIngestionConfig,
    core_cfg: Config,
    force: bool = False,
    workers: int = 0,
) -> str:
    paths = get_assets_paths(config)
    clean_text_dir = paths["clean_text"]
    taxonomy_path = paths["taxonomy"]

    w = resolve_workers(config=config, workers=workers)
    logger.info(
        "taxonomy: source=%s output=%s force=%s workers=%d",
        clean_text_dir,
        taxonomy_path,
        force,
        w,
    )
    build_taxonomy(
        source_root=clean_text_dir,
        output_path=taxonomy_path,
        config=config,
        core_cfg=core_cfg,
        force_rebuild=force,
        workers=w,
    )
    return str(taxonomy_path)


class TaxonomyTransformer(BaseTransformer):
    """Ingestion stage transformer: CleanText -> taxonomy.json."""

    name = "ingestion.taxonomy"

    def __init__(self) -> None:
        super().__init__()
        self._config: RagIngestionConfig | None = None
        self._core_cfg: Config | None = None

    def configure(self, params: dict[str, Any] | None) -> None:
        super().configure(params)
        cfg = (params or {}).get("config")
        if isinstance(cfg, RagIngestionConfig):
            self._config = cfg
        elif isinstance(cfg, dict):
            self._config = RagIngestionConfig.model_validate(cfg)
        else:
            self._config = None
        cc = (params or {}).get("core_cfg")
        self._core_cfg = cc if isinstance(cc, Config) else None

    async def transform(self, unit: ContextUnit) -> ContextUnit:
        unit.provenance.append(self.name)

        payload = unit.payload or {}
        if not isinstance(payload, dict):
            payload = {}

        cfg = None
        content = payload.get("content")
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

        if isinstance(content, RagIngestionConfig):
            cfg = content
        elif isinstance(content, dict):
            cfg = RagIngestionConfig.model_validate(content)
        elif isinstance(metadata.get("ingestion_config"), RagIngestionConfig):
            cfg = metadata["ingestion_config"]
        elif isinstance(metadata.get("ingestion_config"), dict):
            cfg = RagIngestionConfig.model_validate(metadata["ingestion_config"])
        elif isinstance(metadata.get("config"), RagIngestionConfig):
            cfg = metadata["config"]
        elif isinstance(metadata.get("config"), dict):
            cfg = RagIngestionConfig.model_validate(metadata["config"])
        if cfg is None and self._config is not None:
            cfg = self._config
        if cfg is None:
            cfg = load_config()

        overwrite = bool(metadata.get("overwrite", self.params.get("overwrite", True)))
        workers = int(metadata.get("workers", self.params.get("workers", 0)))

        if self._core_cfg is None:
            raise ValueError(
                "TaxonomyTransformer requires core_cfg (Config) via configure(params={'core_cfg': ...})"
            )
        taxonomy_path = build_taxonomy_from_clean_text(
            config=cfg, core_cfg=self._core_cfg, force=overwrite, workers=workers
        )
        metadata["taxonomy_path"] = taxonomy_path
        # Also include resolved assets paths for downstream stages.
        try:
            paths = get_assets_paths(cfg)
            if "assets_paths" not in metadata:
                metadata["assets_paths"] = {}
            metadata["assets_paths"].update({k: str(v) for k, v in paths.items()})
        except Exception as e:
            logger.debug("Failed to get assets paths: %s", e)

        payload["metadata"] = metadata
        unit.payload = payload

        return unit


__all__ = ["build_taxonomy_from_clean_text", "TaxonomyTransformer"]
