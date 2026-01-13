"""Transformer: CleanText + taxonomy + graph -> ShadowRecords (per type)."""

from __future__ import annotations

import logging
import time
from typing import Any

from contextrouter.core import BaseTransformer, BisquitEnvelope, Config
from contextrouter.modules.ingestion.rag.config import get_assets_paths, load_config
from contextrouter.modules.ingestion.rag.core import get_all_plugins
from contextrouter.modules.ingestion.rag.core.utils import parallel_map, resolve_workers
from contextrouter.modules.ingestion.rag.graph.lookup import GraphEnricher
from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig
from contextrouter.modules.ingestion.rag.stages.store import (
    read_raw_data_jsonl,
    write_shadow_records_jsonl,
)

logger = logging.getLogger(__name__)


def build_shadow_records(
    *,
    config: RagIngestionConfig,
    core_cfg: Config,
    only_types: list[str],
    overwrite: bool = True,
    workers: int = 1,
) -> dict[str, str]:
    paths = get_assets_paths(config)
    taxonomy_path = paths["taxonomy"]
    graph_path = paths["graph"]

    enricher = GraphEnricher(graph_path, taxonomy_path=taxonomy_path)

    out_paths: dict[str, str] = {}

    plugins = [p() for p in get_all_plugins()]
    plugins_by_type = {p.source_type: p for p in plugins}

    def _run_one(t: str) -> tuple[str, str]:
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
    name = "ingestion.shadow"

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

    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        envelope.add_trace(self.name)

        cfg = None
        if isinstance(envelope.content, RagIngestionConfig):
            cfg = envelope.content
        elif isinstance(envelope.content, dict):
            cfg = RagIngestionConfig.model_validate(envelope.content)
        elif isinstance(envelope.metadata.get("ingestion_config"), RagIngestionConfig):
            cfg = envelope.metadata["ingestion_config"]
        elif isinstance(envelope.metadata.get("ingestion_config"), dict):
            cfg = RagIngestionConfig.model_validate(envelope.metadata["ingestion_config"])
        elif isinstance(envelope.metadata.get("config"), RagIngestionConfig):
            cfg = envelope.metadata["config"]
        elif isinstance(envelope.metadata.get("config"), dict):
            cfg = RagIngestionConfig.model_validate(envelope.metadata["config"])
        if cfg is None and self._config is not None:
            cfg = self._config
        if cfg is None:
            cfg = load_config()
        if self._core_cfg is None:
            raise ValueError(
                "ShadowTransformer requires core_cfg (Config) via configure(params={'core_cfg': ...})"
            )

        only_types = envelope.metadata.get("only_types")
        if not isinstance(only_types, list) or not only_types:
            only_types = ["video", "book", "qa", "web", "knowledge"]
        only_types = [t for t in only_types if isinstance(t, str) and t.strip()]

        overwrite = bool(envelope.metadata.get("overwrite", self.params.get("overwrite", True)))
        workers = int(envelope.metadata.get("workers", self.params.get("workers", 1)))

        out = build_shadow_records(
            config=cfg,
            core_cfg=self._core_cfg,
            only_types=only_types,
            overwrite=overwrite,
            workers=workers,
        )
        envelope.metadata["shadow_paths"] = out
        return envelope


__all__ = ["build_shadow_records", "ShadowTransformer"]
