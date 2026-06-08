"""Transformer: CleanText + taxonomy + graph -> ShadowRecords (per type)."""

from __future__ import annotations

import time
from typing import override

from contextunity.brain.core.config import BrainConfig
from contextunity.brain.ingestion.rag.config import get_assets_paths, load_config
from contextunity.brain.ingestion.rag.core import get_all_plugins
from contextunity.brain.ingestion.rag.core.utils import parallel_map, resolve_workers
from contextunity.brain.ingestion.rag.graph.lookup import GraphEnricher
from contextunity.brain.ingestion.rag.settings import RagIngestionConfig
from contextunity.brain.ingestion.rag.stages.store import (
    read_raw_data_jsonl,
    write_shadow_records_jsonl,
)
from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.sdk.payload import get_bool, get_int, get_str_list
from contextunity.core.types import JsonDict

from contextunity.router.core import BaseTransformer, ContextUnit
from contextunity.router.modules.transformers._ingestion_helpers import (
    payload_metadata,
    resolve_rag_ingestion_config,
)

logger = get_contextunit_logger(__name__)


def build_shadow_records(
    *,
    config: RagIngestionConfig,
    core_cfg: BrainConfig,
    only_types: list[str],
    overwrite: bool = True,
    workers: int = 1,
) -> dict[str, str]:
    """Run all source-type plugins in parallel to produce shadow JSONL files.

    For each type in *only_types*, loads the clean-text JSONL, runs the matching
    plugin's ``transform`` (with graph enrichment from taxonomy + graph assets),
    and writes the output to ``<shadow>/<type>.jsonl``.

    Returns a ``{source_type: output_path}`` mapping for types that produced output.
    """
    paths = get_assets_paths(config)
    taxonomy_path = paths["taxonomy"]
    graph_path = paths["graph"]

    enricher = GraphEnricher(graph_path, taxonomy_path=taxonomy_path)

    out_paths: dict[str, str] = {}

    plugins = [p() for p in get_all_plugins()]
    plugins_by_type = {p.source_type: p for p in plugins}

    def _run_one(t: str) -> tuple[str, str]:
        """Process a single source type: read clean-text JSONL, transform via plugin, write shadow records."""
        t0 = time.perf_counter()
        plugin = plugins_by_type.get(t)
        if plugin is None:
            logger.warning("shadow: no plugin registered for type=%s (skipping)", t)
            return (t, "")

        in_path = paths["clean_text"] / f"{t}.jsonl"
        items = read_raw_data_jsonl(in_path)
        if not items:
            logger.warning("shadow: no clean_text items for type=%s at %s", t, in_path)
            return (t, "")

        records = plugin.transform(
            items,
            enrichment_func=enricher.get_context,
            taxonomy_path=taxonomy_path,
            config=config,
            core_cfg=core_cfg,
        )

        out_path = paths["shadow"] / f"{t}.jsonl"
        count = write_shadow_records_jsonl(records, out_path, overwrite=overwrite)
        logger.warning(
            "shadow: wrote %d records for type=%s -> %s (%.1fs)",
            count,
            t,
            out_path,
            time.perf_counter() - t0,
        )
        return (t, str(out_path))

    w = resolve_workers(config=config, workers=workers)
    results = parallel_map(only_types, _run_one, workers=w, ordered=False, swallow_exceptions=False)
    for r in results:
        if not r:
            continue
        tt, out_path = r
        if out_path:
            out_paths[tt] = out_path

    return out_paths


class ShadowTransformer(BaseTransformer):
    """Ingestion-stage transformer: wraps ``build_shadow_records`` as a ContextUnit pipeline step."""

    name: str = "ingestion.shadow"

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
        run ``build_shadow_records``, and store output paths in ``metadata.shadow_paths``.
        """
        unit.provenance.append(self.name)

        payload, metadata = payload_metadata(unit.payload)

        cfg = resolve_rag_ingestion_config(
            payload,
            metadata,
            configured=self._config,
            default_loader=load_config,
        )
        if self._core_cfg is None:
            raise ConfigurationError(
                "ShadowTransformer requires core_cfg (Config) via configure(params={'core_cfg': ...})"
            )

        only_types = get_str_list(metadata, "only_types")
        if not only_types:
            only_types = ["video", "book", "qa", "web", "knowledge"]

        overwrite = get_bool(metadata, "overwrite", bool(self.params.get("overwrite", True)))
        workers = get_int(metadata, "workers", get_int(self.params, "workers", 1))

        out = build_shadow_records(
            config=cfg,
            core_cfg=self._core_cfg,
            only_types=only_types,
            overwrite=overwrite,
            workers=workers,
        )
        shadow_paths: JsonDict = {k: v for k, v in out.items()}
        metadata["shadow_paths"] = shadow_paths
        payload["metadata"] = metadata
        unit.payload = payload
        return unit


__all__ = ["build_shadow_records", "ShadowTransformer"]
