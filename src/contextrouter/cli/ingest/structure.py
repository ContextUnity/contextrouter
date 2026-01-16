"""Structure command: CleanText → taxonomy.json + ontology.json."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    coerce_types,
    load_core_cfg,
    require_clean_text,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    build_ingestion_report,
    build_ontology_from_taxonomy,
    build_taxonomy_from_clean_text,
    get_assets_paths,
    load_config,
    resolve_workers,
)

logger = logging.getLogger(__name__)


@click.command("structure")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--force", is_flag=True, default=False)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
@click.option("--workers", type=int, default=0, show_default="auto")
@click.option("--skip-taxonomy", is_flag=True, default=False)
@click.option("--skip-ontology", is_flag=True, default=False)
def cmd_structure(
    config_path: Path | None,
    force: bool,
    only_types: tuple[str, ...],
    workers: int,
    skip_taxonomy: bool,
    skip_ontology: bool,
) -> None:
    """Build taxonomy.json and ontology.json from CleanText."""
    suppress_noisy_loggers()
    stage_banner("STRUCTURE (CleanText → taxonomy.json + ontology.json)")

    cfg = load_config(config_path)
    core_cfg = load_core_cfg(cfg)
    paths = get_assets_paths(cfg)

    types = coerce_types(only_types)
    require_clean_text(paths, types)

    w = resolve_workers(config=cfg, workers=workers)
    click.echo(f"  Workers: {w}")

    # Build taxonomy if not skipped
    if not skip_taxonomy:
        click.echo("  Building taxonomy...")
        build_taxonomy_from_clean_text(config=cfg, core_cfg=core_cfg, force=force, workers=w)
        click.echo(click.style("  ✓ taxonomy completed", fg="green"))
    else:
        click.echo("  Skipping taxonomy (already exists)")

    # Build ontology if not skipped
    if not skip_ontology:
        click.echo("  Building ontology...")
        build_ontology_from_taxonomy(config=cfg, overwrite=force)
        click.echo(click.style("  ✓ ontology completed", fg="green"))
    else:
        click.echo("  Skipping ontology (already exists)")

    click.echo(click.style("✓ structure completed", fg="green"))
    try:
        build_ingestion_report(config=cfg, types=types)
    except Exception:
        logger.debug("report: failed after structure", exc_info=True)


__all__ = ["cmd_structure"]
