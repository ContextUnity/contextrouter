"""Report command: Generate ingestion report."""

from __future__ import annotations

from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    coerce_types,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    build_ingestion_report,
    get_assets_paths,
    load_config,
)


@click.command("report")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
def cmd_report(config_path: Path | None, only_types: tuple[str, ...]) -> None:
    """Build ingestion_report.json from current artifacts."""
    suppress_noisy_loggers()
    stage_banner("REPORT (Artifacts → ingestion_report.json)")
    cfg = load_config(config_path)
    _ = get_assets_paths(cfg)
    types = coerce_types(only_types)
    click.echo(f"  Types: {', '.join(types)}")
    out = build_ingestion_report(config=cfg, types=types)
    click.echo(click.style(f"✓ report completed ({out})", fg="green"))


__all__ = ["cmd_report"]
