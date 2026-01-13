"""Export command: ShadowRecords → JSONL per type."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    coerce_types,
    require_shadow,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    build_ingestion_report,
    export_jsonl_per_type,
    get_assets_paths,
    load_config,
    resolve_workers,
)

logger = logging.getLogger(__name__)


@click.command("export")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
@click.option("--overwrite/--no-overwrite", default=True)
@click.option("--workers", type=int, default=0, show_default="auto")
def cmd_export(
    config_path: Path | None, only_types: tuple[str, ...], overwrite: bool, workers: int
) -> None:
    """Export ShadowRecords to Vertex import JSONL format."""
    suppress_noisy_loggers()
    stage_banner("EXPORT (ShadowRecords → JSONL per type)")

    t0 = time.perf_counter()
    cfg = load_config(config_path)
    paths = get_assets_paths(cfg)

    types = coerce_types(only_types)
    click.echo(f"  Types: {', '.join(types)}")
    require_shadow(paths, types)

    w = resolve_workers(config=cfg, workers=workers)
    click.echo(f"  Overwrite: {overwrite}")
    click.echo(f"  Workers: {w}")
    export_jsonl_per_type(config=cfg, only_types=types, overwrite=overwrite, workers=w)
    click.echo(click.style(f"✓ export completed ({time.perf_counter() - t0:.1f}s)", fg="green"))
    try:
        build_ingestion_report(config=cfg, types=types)
    except Exception:
        logger.debug("report: failed after export", exc_info=True)


__all__ = ["cmd_export"]
