"""Index command: CleanText + taxonomy + ontology → graph.pickle + shadow records."""

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
    build_graph_from_clean_text,
    build_ingestion_report,
    build_shadow_records,
    get_assets_paths,
    load_config,
    resolve_workers,
)

logger = logging.getLogger(__name__)


@click.command("index")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--force", is_flag=True, default=False)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
@click.option("--workers", type=int, default=0, show_default="auto")
@click.option("--skip-graph", is_flag=True, default=False)
@click.option("--skip-shadow", is_flag=True, default=False)
def cmd_index(
    config_path: Path | None,
    force: bool,
    only_types: tuple[str, ...],
    workers: int,
    skip_graph: bool,
    skip_shadow: bool,
) -> None:
    """Build knowledge graph and shadow records from processed data."""
    suppress_noisy_loggers()
    stage_banner("INDEX (CleanText + taxonomy + ontology → graph.pickle + shadow records)")

    cfg = load_config(config_path)
    core_cfg = load_core_cfg(cfg)
    paths = get_assets_paths(cfg)

    types = coerce_types(only_types)
    require_clean_text(paths, types)

    w = resolve_workers(config=cfg, workers=workers)
    click.echo(f"  Workers: {w}")

    # Build graph if not skipped
    if not skip_graph:
        click.echo("  Building knowledge graph...")
        build_graph_from_clean_text(config=cfg, core_cfg=core_cfg, workers=w, overwrite=force)
        click.echo(click.style("  ✓ graph completed", fg="green"))
    else:
        click.echo("  Skipping graph (already exists)")

    # Build shadow records if not skipped
    if not skip_shadow:
        click.echo("  Building shadow records...")
        build_shadow_records(
            config=cfg,
            core_cfg=core_cfg,
            only_types=list(types),
            overwrite=force,
            workers=w if w > 0 else 1,
        )
        click.echo(click.style("  ✓ shadow records completed", fg="green"))
    else:
        click.echo("  Skipping shadow records (already exist)")

    click.echo(click.style("✓ index completed", fg="green"))
    try:
        build_ingestion_report(config=cfg, types=types)
    except Exception:
        logger.debug("report: failed after index", exc_info=True)


__all__ = ["cmd_index"]
