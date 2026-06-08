"""Transformer: taxonomy.json -> ontology.json (schema-first)."""

from __future__ import annotations

from pathlib import Path
from typing import override

from contextunity.brain.ingestion.rag.config import get_assets_paths, load_config
from contextunity.brain.ingestion.rag.graph.builder import ALLOWED_RELATION_LABELS
from contextunity.brain.ingestion.rag.settings import RagIngestionConfig
from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps
from contextunity.core.sdk.payload import get_bool

from contextunity.router.core import BaseTransformer, ContextUnit
from contextunity.router.modules.transformers._ingestion_helpers import (
    payload_metadata,
    resolve_rag_ingestion_config,
)

logger = get_contextunit_logger(__name__)

_DEFAULT_ENTITY_TYPES: list[str] = [
    "Concept",
    "Principle",
    "Practice",
    "Person",
    "Resource",
    "Event",
]


def build_ontology_from_taxonomy(*, config: RagIngestionConfig, overwrite: bool = True) -> Path:
    """Generate ``ontology.json`` containing entity type and relation label whitelists
    derived from the graph builder's ``ALLOWED_RELATION_LABELS``.

    Skips writing if the file already exists and *overwrite* is ``False``.
    Returns the output path.
    """
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
    _ = ontology_path.write_text(json_dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("ontology: wrote %s", ontology_path)
    return ontology_path


class OntologyTransformer(BaseTransformer):
    """Ingestion stage transformer: taxonomy.json -> ontology.json."""

    name: str = "ingestion.ontology"

    def __init__(self) -> None:
        """Initialize the ``RagIngestionConfig`` slot — populated later via ``configure()``."""
        super().__init__()
        self._config: RagIngestionConfig | None = None

    @override
    def configure(self, params: dict[str, object] | None) -> None:
        """Extract ``RagIngestionConfig`` from *params* for later use in ``transform``."""
        super().configure(params)
        cfg = (params or {}).get("config")
        if isinstance(cfg, RagIngestionConfig):
            self._config = cfg
        elif isinstance(cfg, dict):
            self._config = RagIngestionConfig.model_validate(cfg)
        else:
            self._config = None

    @override
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """Resolve ingestion config (payload → metadata → self._config → default),
        run ``build_ontology_from_taxonomy``, and store the output path in ``metadata.ontology_path``.
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
        out = build_ontology_from_taxonomy(config=cfg, overwrite=overwrite)
        metadata["ontology_path"] = str(out)
        payload["metadata"] = metadata
        unit.payload = payload
        return unit


__all__ = ["build_ontology_from_taxonomy", "OntologyTransformer"]
