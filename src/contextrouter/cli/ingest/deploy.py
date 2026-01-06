"""Deploy command: JSONL → Cloud Search Index."""

from __future__ import annotations

from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    coerce_types,
    require_exports,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    deploy_jsonl_files,
    get_assets_paths,
    load_config,
)


@click.command("deploy")
@click.option(
    "--data-store-id",
    type=str,
    default=None,
    help="Override data_store_id (can be 'blue', 'green', or actual ID)",
)
@click.option("--wait/--no-wait", default=False)
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
def cmd_deploy(
    data_store_id: str | None,
    wait: bool,
    config_path: Path | None,
    only_types: tuple[str, ...],
) -> None:
    """Deploy JSONL exports to configured search index (Vertex AI Search, Postgres, etc.)."""
    suppress_noisy_loggers()
    stage_banner("DEPLOY (JSONL → Cloud Search Index)")

    cfg = load_config(config_path)
    paths = get_assets_paths(cfg)

    # CLI override: inject data_store_id into config if provided
    if data_store_id:
        cfg.setdefault("upload", {}).setdefault("gcloud", {})["data_store_id"] = data_store_id
        click.echo(f"  Override: data_store_id={data_store_id}")

    types = coerce_types(only_types)
    click.echo(f"  Types: {', '.join(types)}")
    require_exports(paths, types)

    # Pick latest export per type by mtime
    jsonl_paths: dict[str, str] = {}
    for t in types:
        d = paths["jsonl"] / t
        candidates = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            jsonl_paths[t] = str(candidates[0])

    results = deploy_jsonl_files(jsonl_paths_by_type=jsonl_paths, config=cfg, wait=wait)

    # Report results
    failed = [t for t, r in results.items() if not r.success]
    if failed:
        click.echo(click.style(f"✗ deploy failed for: {', '.join(failed)}", fg="red"))
        raise SystemExit(1)

    click.echo(click.style("✓ deploy completed", fg="green"))


__all__ = ["cmd_deploy"]
