"""Preprocess command: Raw → CleanText."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    coerce_types,
    load_core_cfg,
    stage_banner,
    suppress_noisy_loggers,
)

logger = logging.getLogger(__name__)


@click.command("preprocess")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
@click.option("--overwrite/--no-overwrite", default=True)
@click.option("--workers", type=int, default=0, show_default="auto")
def cmd_preprocess(
    config_path: Path | None, only_types: tuple[str, ...], overwrite: bool, workers: int
) -> None:
    """Preprocess raw source files into CleanText JSONL."""
    # Import heavy ingestion modules lazily so `--help` works without extras installed.
    from contextrouter.modules.ingestion.rag import (
        ensure_directories_exist,
        get_assets_paths,
        load_config,
        preprocess_to_clean_text,
        resolve_workers,
    )

    suppress_noisy_loggers()
    stage_banner("PREPROCESS (Raw → CleanText)")

    t0 = time.perf_counter()
    cfg = load_config(config_path)
    core_cfg = load_core_cfg(cfg)
    paths = get_assets_paths(cfg)
    ensure_directories_exist(paths)

    types = coerce_types(only_types)
    click.echo(f"  Types: {', '.join(types)}")
    click.echo(f"  Overwrite: {overwrite}")
    w = resolve_workers(config=cfg, workers=workers)
    click.echo(f"  Workers: {w}")
    preprocess_to_clean_text(
        core_cfg=core_cfg, config=cfg, only_types=types, overwrite=overwrite, workers=w
    )
    click.echo(click.style(f"✓ preprocess completed ({time.perf_counter() - t0:.1f}s)", fg="green"))


__all__ = ["cmd_preprocess"]
