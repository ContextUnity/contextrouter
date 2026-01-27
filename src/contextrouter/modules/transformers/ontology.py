"""Transformer: taxonomy.json -> ontology.json (schema-first)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from contextrouter.core import BaseTransformer, ContextUnit
from contextrouter.modules.ingestion.rag.config import get_assets_paths, load_config
from contextrouter.modules.ingestion.rag.graph.builder import ALLOWED_RELATION_LABELS
from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig

logger = logging.getLogger(__name__)


_DEFAULT_ENTITY_TYPES: list[str] = [
    "Concept",
    "Principle",
    "Practice",
    "Person",
    "Resource",
    "Event",
]


def build_ontology_from_taxonomy(*, config: RagIngestionConfig, overwrite: bool = True) -> Path:
    """Build ontology.json, derived from the project’s current relation label whitelist."""
    paths = get_assets_paths(config)
    taxonomy_path = paths["taxonomy"]
    ontology_path = paths["ontology"]

    if ontology_path.exists() and not overwrite:
        return ontology_path

    # Allowed labels should stay aligned with graph extraction whitelist.
    allowed_labels = sorted(list(ALLOWED_RELATION_LABELS))

    # Conservative: only a subset is used as runtime facts to avoid noisy spam.
    runtime_fact_labels = [
        "CAUSES",
        "LEADS_TO",
        "ENABLES",
        "REQUIRES",
        "SUPPORTS",
        "OPPOSES",
        "PREVENTS",
        "RESULTS_IN",
        "IS_A",
        "IS_PART_OF",
        "INCLUDES",
        "EXAMPLE_OF",
        "APPLIES_TO",
        "IS_ABOUT",
    ]
    runtime_fact_labels = [x for x in runtime_fact_labels if x in ALLOWED_RELATION_LABELS]

    payload = {
        "version": "1.0",
        "entity_types": _DEFAULT_ENTITY_TYPES,
        "relations": {
            "allowed_labels": allowed_labels,
            "runtime_fact_labels": runtime_fact_labels,
            "notes": "Graph storage is currently undirected; labels may be directional but are treated as hints.",
        },
        "constraints": {
            "max_entity_label_chars": 120,
            "min_entity_label_chars": 2,
        },
        "inputs": {
            "taxonomy_path": str(taxonomy_path),
        },
    }

    ontology_path.parent.mkdir(parents=True, exist_ok=True)
    ontology_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ontology: wrote %s", ontology_path)
    return ontology_path


class OntologyTransformer(BaseTransformer):
    """Ingestion stage transformer: taxonomy.json -> ontology.json."""

    name = "ingestion.ontology"

    def __init__(self) -> None:
        super().__init__()
        self._config: RagIngestionConfig | None = None

    def configure(self, params: dict[str, Any] | None) -> None:
        super().configure(params)
        cfg = (params or {}).get("config")
        if isinstance(cfg, RagIngestionConfig):
            self._config = cfg
        elif isinstance(cfg, dict):
            self._config = RagIngestionConfig.model_validate(cfg)
        else:
            self._config = None

    async def transform(self, unit: ContextUnit) -> ContextUnit:
        unit.provenance.append(self.name)

        payload = unit.payload or {}
        if not isinstance(payload, dict):
            payload = {}
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

        cfg = None
        if isinstance(payload.get("content"), RagIngestionConfig):
            cfg = payload.get("content")
        elif isinstance(payload.get("content"), dict):
            cfg = RagIngestionConfig.model_validate(payload.get("content"))
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
        out = build_ontology_from_taxonomy(config=cfg, overwrite=overwrite)
        metadata["ontology_path"] = str(out)
        payload["metadata"] = metadata
        unit.payload = payload
        return unit


__all__ = ["build_ontology_from_taxonomy", "OntologyTransformer"]
