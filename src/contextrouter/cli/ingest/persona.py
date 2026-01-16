"""Persona command: CleanText → persona.txt."""

from __future__ import annotations

from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    load_core_cfg,
    require_clean_text,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    build_persona,
    get_assets_paths,
    load_config,
)


@click.command("persona")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
def cmd_persona(config_path: Path | None) -> None:
    """Generate persona.txt (optional; controlled by [persona].enabled)."""
    suppress_noisy_loggers()
    stage_banner("PERSONA (CleanText → persona.txt)")
    cfg = load_config(config_path)
    core_cfg = load_core_cfg(cfg)
    paths = get_assets_paths(cfg)
    require_clean_text(paths, list(ALL_TYPES))
    out = build_persona(config=cfg, core_cfg=core_cfg)
    if out is None:
        click.echo(click.style("⊘ persona disabled (persona.enabled=false)", fg="yellow"))
        return
    click.echo(click.style(f"✓ persona completed ({out})", fg="green"))


__all__ = ["cmd_persona"]
