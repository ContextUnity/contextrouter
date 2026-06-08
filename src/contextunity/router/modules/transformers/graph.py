"""Transformer: CleanText + taxonomy.json -> knowledge_graph.pickle."""

from __future__ import annotations

from pathlib import Path
from typing import override

from contextunity.brain.core.config import BrainConfig
from contextunity.brain.ingestion.rag.config import get_assets_paths, load_config
from contextunity.brain.ingestion.rag.core import RawData
from contextunity.brain.ingestion.rag.core.utils import resolve_workers
from contextunity.brain.ingestion.rag.graph.builder import GraphBuilder
from contextunity.brain.ingestion.rag.settings import RagIngestionConfig
from contextunity.brain.ingestion.rag.stages.store import read_raw_data_jsonl
from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.sdk.payload import get_bool, get_int

from contextunity.router.core import BaseTransformer, ContextUnit
from contextunity.router.modules.transformers._ingestion_helpers import (
    payload_metadata,
    resolve_rag_ingestion_config,
)

logger = get_contextunit_logger(__name__)


def build_graph_from_clean_text(
    *,
    config: RagIngestionConfig,
    core_cfg: BrainConfig,
    workers: int = 0,
    overwrite: bool = True,
) -> str:
    """Aggregate clean-text JSONL for each source type, run ``GraphBuilder``
    (LLM-based entity/relation extraction) with taxonomy and optional ontology
    constraints, and write the resulting knowledge graph pickle.

    Returns the output graph path as a string.
    """
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

    all_items: list[RawData] = []
    for t in include_types:
        if not t.strip():
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

    if isinstance(ontology_path, Path):
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
    """Ingestion-stage transformer: wraps ``build_graph_from_clean_text`` as a ContextUnit pipeline step."""

    name: str = "ingestion.graph"

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
        run ``build_graph_from_clean_text``, and store the output path in ``metadata.graph_path``.

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
        if self._core_cfg is None:
            raise ConfigurationError(
                "GraphTransformer requires core_cfg (Config) via configure(params={'core_cfg': ...})"
            )

        overwrite = get_bool(metadata, "overwrite", bool(self.params.get("overwrite", True)))
        workers = get_int(metadata, "workers", get_int(self.params, "workers", 0))
        out = build_graph_from_clean_text(
            config=cfg, core_cfg=self._core_cfg, workers=workers, overwrite=overwrite
        )
        metadata["graph_path"] = out
        payload["metadata"] = metadata
        unit.payload = payload
        return unit


__all__ = ["build_graph_from_clean_text", "GraphTransformer"]
