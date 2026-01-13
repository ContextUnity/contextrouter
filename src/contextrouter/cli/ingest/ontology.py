"""Ontology command: taxonomy.json → ontology.json."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    require_taxonomy,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    build_ingestion_report,
    build_ontology_from_taxonomy,
    get_assets_paths,
    load_config,
)

logger = logging.getLogger(__name__)


@click.command("ontology")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--overwrite/--no-overwrite", default=True)
def cmd_ontology(config_path: Path | None, overwrite: bool) -> None:
    """Build ontology.json from current ingestion contracts."""
    suppress_noisy_loggers()
    stage_banner("ONTOLOGY (taxonomy.json → ontology.json)")
    cfg = load_config(config_path)
    paths = get_assets_paths(cfg)
    require_taxonomy(paths)
    out = build_ontology_from_taxonomy(config=cfg, overwrite=overwrite)
    click.echo(click.style(f"✓ ontology completed ({out})", fg="green"))
    try:
        build_ingestion_report(config=cfg, types=list(ALL_TYPES))
    except Exception:
        logger.debug("report: failed after ontology", exc_info=True)


__all__ = ["cmd_ontology"]
