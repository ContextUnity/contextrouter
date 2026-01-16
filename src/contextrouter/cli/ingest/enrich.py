"""Enrich command: CleanText → NER + keyphrases."""

from __future__ import annotations

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
from contextrouter.modules.ingestion.rag import enrich_clean_text, get_assets_paths, load_config


@click.command("enrich")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
@click.option("--overwrite/--no-overwrite", default=True)
@click.option("--workers", type=int, default=0, show_default="auto")
def cmd_enrich(
    config_path: Path | None,
    only_types: tuple[str, ...],
    overwrite: bool,
    workers: int,
) -> None:
    """Enrich CleanText with NER + keyphrases (config‑gated)."""
    suppress_noisy_loggers()
    stage_banner("ENRICH (CleanText → NER + keyphrases) (optional)")

    cfg = load_config(config_path)
    core_cfg = load_core_cfg(cfg)
    paths = get_assets_paths(cfg)

    types = coerce_types(only_types)
    require_clean_text(paths, types)

    out = enrich_clean_text(
        config=cfg, core_cfg=core_cfg, only_types=types, overwrite=overwrite, workers=workers
    )
    if not out:
        click.echo(click.style("  ⊘ enrichment skipped (disabled or empty)", fg="yellow"))
        return
    click.echo(click.style("  ✓ enrichment done", fg="green"))


__all__ = ["cmd_enrich"]
