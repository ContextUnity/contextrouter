"""Transformer: CleanText + taxonomy.json -> knowledge_graph.pickle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from contextrouter.core import BaseTransformer, BisquitEnvelope, Config
from contextrouter.modules.ingestion.rag.config import get_assets_paths, load_config
from contextrouter.modules.ingestion.rag.core.utils import resolve_workers
from contextrouter.modules.ingestion.rag.graph.builder import GraphBuilder
from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig
from contextrouter.modules.ingestion.rag.stages.store import read_raw_data_jsonl

logger = logging.getLogger(__name__)


def build_graph_from_clean_text(
    *,
    config: RagIngestionConfig,
    core_cfg: Config,
    workers: int = 0,
    overwrite: bool = True,
) -> str:
    paths = get_assets_paths(config)
    clean_text_dir = paths["clean_text"]
    taxonomy_path = paths["taxonomy"]
    graph_path = paths["graph"]
    ontology_path = paths.get("ontology")

    include_types = config.graph.include_types
    if not include_types:
        include_types = ["video", "book", "qa", "knowledge"]

    incremental = config.graph.incremental
    if overwrite:
        incremental = False
    model = config.graph.model.strip() or core_cfg.models.ingestion.graph.model.strip()
    builder_mode = config.graph.builder_mode

    all_items: list[Any] = []
    for t in include_types:
        if not isinstance(t, str) or not t.strip():
            continue
        p = clean_text_dir / f"{t}.jsonl"
        all_items.extend(read_raw_data_jsonl(p))

    logger.info(
        "graph: source=%s include_types=%s items=%d output=%s workers=%d overwrite=%s incremental=%s model=%s",
        clean_text_dir,
        include_types,
        len(all_items),
        graph_path,
        workers,
        overwrite,
        incremental,
        model,
    )

    if ontology_path and isinstance(ontology_path, Path):
        if ontology_path.exists():
            logger.info("graph: using ontology from %s", ontology_path)
        else:
            logger.warning("graph: ontology path specified but file not found: %s", ontology_path)
    else:
        logger.info("graph: no ontology specified, using default allowed_labels")

    w = resolve_workers(config=config, workers=workers)
    builder = GraphBuilder(
        max_workers=w,
        taxonomy_path=taxonomy_path,
        ontology_path=ontology_path if isinstance(ontology_path, Path) else None,
        model=model,
        mode=builder_mode,
        core_cfg=core_cfg,
    )
    builder.build(all_items, graph_path, incremental=incremental)
    return str(graph_path)


class GraphTransformer(BaseTransformer):
    name = "ingestion.graph"

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
                "GraphTransformer requires core_cfg (Config) via configure(params={'core_cfg': ...})"
            )

        overwrite = bool(envelope.metadata.get("overwrite", self.params.get("overwrite", True)))
        workers = int(envelope.metadata.get("workers", self.params.get("workers", 0)))
        out = build_graph_from_clean_text(
            config=cfg, core_cfg=self._core_cfg, workers=workers, overwrite=overwrite
        )
        envelope.metadata["graph_path"] = out
        return envelope


__all__ = ["build_graph_from_clean_text", "GraphTransformer"]
