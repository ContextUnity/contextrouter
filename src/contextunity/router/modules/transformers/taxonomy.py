"""Transformer: CleanText -> taxonomy.json."""

from __future__ import annotations

from typing import override

from contextunity.brain.core.config import BrainConfig
from contextunity.brain.ingestion.rag.config import get_assets_paths, load_config
from contextunity.brain.ingestion.rag.core.utils import resolve_workers
from contextunity.brain.ingestion.rag.processors.taxonomy_builder import (
    build_taxonomy,
)
from contextunity.brain.ingestion.rag.settings import RagIngestionConfig
from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.sdk.payload import get_bool, get_int
from contextunity.core.types import is_json_dict

from contextunity.router.core import BaseTransformer, ContextUnit
from contextunity.router.modules.transformers._ingestion_helpers import (
    payload_metadata,
    resolve_rag_ingestion_config,
)

logger = get_contextunit_logger(__name__)


def build_taxonomy_from_clean_text(
    *,
    config: RagIngestionConfig,
    core_cfg: BrainConfig,
    force: bool = False,
    workers: int = 0,
) -> str:
    """Delegate to ``build_taxonomy`` to extract a concept hierarchy from clean-text JSONL.

    Returns the output taxonomy path as a string.
    """
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
    _ = build_taxonomy(
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

    name: str = "ingestion.taxonomy"

    def __init__(self) -> None:
        """Initialize config slots — populated later via ``configure()``."""
        super().__init__()
        self._config: RagIngestionConfig | None = None
        self._core_cfg: BrainConfig | None = None

    @override
    def configure(self, params: dict[str, object] | None) -> None:
        """Extract ``RagIngestionConfig`` and ``BrainConfig`` from *params* for later use in ``transform``."""
        super().configure(params)
        cfg = (params or {}).get("config")
        if isinstance(cfg, RagIngestionConfig):
            self._config = cfg
        elif isinstance(cfg, dict):
            self._config = RagIngestionConfig.model_validate(cfg)
        else:
            self._config = None
        cc = (params or {}).get("core_cfg")
        self._core_cfg = cc if isinstance(cc, BrainConfig) else None

    @override
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """Resolve ingestion config (payload → metadata → self._config → default),
        run ``build_taxonomy_from_clean_text``, and store ``taxonomy_path`` and
        resolved ``assets_paths`` in metadata.

        Raises ``ValueError`` if ``core_cfg`` was not provided via ``configure()``.
        """
        unit.provenance.append(self.name)

        payload, metadata = payload_metadata(unit.payload)

        cfg = resolve_rag_ingestion_config(
            payload,
            metadata,
            configured=self._config,
            default_loader=load_config,
        )

        overwrite = get_bool(metadata, "overwrite", bool(self.params.get("overwrite", True)))
        workers = get_int(metadata, "workers", get_int(self.params, "workers", 0))

        if self._core_cfg is None:
            raise ConfigurationError(
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
            assets = metadata["assets_paths"]
            if is_json_dict(assets):
                assets.update({k: str(v) for k, v in paths.items()})
        except Exception as e:
            logger.debug("Failed to get assets paths: %s", e)

        payload["metadata"] = metadata
        unit.payload = payload

        return unit


__all__ = ["build_taxonomy_from_clean_text", "TaxonomyTransformer"]
