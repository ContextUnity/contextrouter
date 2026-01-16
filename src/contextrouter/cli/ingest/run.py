"""Run command: Execute full ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from contextrouter.cli.ingest.common import (
    ALL_TYPES,
    coerce_types,
    load_core_cfg,
    require_clean_text,
    require_exports,
    require_graph,
    require_shadow,
    require_taxonomy,
    stage_banner,
    suppress_noisy_loggers,
)
from contextrouter.modules.ingestion.rag import (
    build_graph_from_clean_text,
    build_ontology_from_taxonomy,
    build_persona,
    build_shadow_records,
    build_taxonomy_from_clean_text,
    enrich_clean_text,
    export_jsonl_per_type,
    get_assets_paths,
    load_config,
    preprocess_to_clean_text,
    resolve_workers,
)


@click.command("run")
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--type", "only_types", multiple=True, type=click.Choice(ALL_TYPES))
@click.option("--skip-preprocess", is_flag=True, default=False)
@click.option("--skip-structure", is_flag=True, default=False)
@click.option("--skip-index", is_flag=True, default=False)
@click.option("--skip-enrich", is_flag=True, default=False)
@click.option("--skip-export", is_flag=True, default=False)
@click.option("--overwrite/--no-overwrite", default=True)
@click.option("--workers", type=int, default=0, show_default="auto")
def cmd_run(
    config_path: Path | None,
    only_types: tuple[str, ...],
    skip_preprocess: bool,
    skip_structure: bool,
    skip_index: bool,
    skip_export: bool,
    skip_enrich: bool,
    overwrite: bool,
    workers: int,
) -> None:
    """Run full ingestion pipeline: preprocess → persona → structure → enrich → index → export."""
    suppress_noisy_loggers()

    click.echo(click.style("\n" + "═" * 60, fg="bright_blue", bold=True))
    click.echo(click.style("  INGESTION PIPELINE", fg="bright_blue", bold=True))
    click.echo(click.style("═" * 60, fg="bright_blue", bold=True))

    cfg = load_config(config_path)
    core_cfg = load_core_cfg(cfg)
    paths = get_assets_paths(cfg)

    types = coerce_types(only_types)
    w = resolve_workers(config=cfg, workers=workers)
    click.echo(f"  Types: {', '.join(types)}")
    click.echo(f"  Overwrite: {overwrite}")
    click.echo(f"  Workers: {w}")

    # Stage 1: Preprocess
    if not skip_preprocess:
        stage_banner("1/5 PREPROCESS (Raw → CleanText)")
        preprocess_to_clean_text(
            core_cfg=core_cfg, config=cfg, only_types=types, overwrite=overwrite, workers=w
        )
        click.echo(click.style("  ✓ preprocess done", fg="green"))
    else:
        click.echo(click.style("  ⊘ preprocess skipped (using existing CleanText)", fg="yellow"))
        require_clean_text(paths, types)

    # Stage 1b: Persona (optional; gated by persona.enabled)
    stage_banner("1b/4 PERSONA (CleanText → persona.txt) (optional)")
    out = build_persona(config=cfg, core_cfg=core_cfg)
    if out is None:
        click.echo(click.style("  ⊘ persona disabled (persona.enabled=false)", fg="yellow"))
    else:
        click.echo(click.style(f"  ✓ persona done ({out})", fg="green"))

    # Stage 2: Structure (taxonomy + ontology)
    if not skip_structure:
        stage_banner("2/4 STRUCTURE (CleanText → taxonomy.json + ontology.json)")
        # Build taxonomy
        build_taxonomy_from_clean_text(config=cfg, core_cfg=core_cfg, force=overwrite, workers=w)
        click.echo(click.style("  ✓ taxonomy done", fg="green"))
        # Build ontology
        try:
            out = build_ontology_from_taxonomy(config=cfg, overwrite=overwrite)
            click.echo(click.style(f"  ✓ ontology done ({out})", fg="green"))
        except Exception as e:
            click.echo(click.style(f"  ⊘ ontology failed ({e})", fg="yellow"))
    else:
        click.echo(
            click.style(
                "  ⊘ structure skipped (using existing taxonomy.json + ontology.json)", fg="yellow"
            )
        )
        require_taxonomy(paths)

    # Stage 2b: Enrichment (optional; independent from structure)
    if not skip_enrich:
        stage_banner("2b/4 ENRICH (CleanText → NER + keyphrases) (optional)")
        import asyncio

        out = asyncio.run(
            enrich_clean_text(
                config=cfg, core_cfg=core_cfg, only_types=types, overwrite=overwrite, workers=w
            )
        )
        if not out:
            click.echo(
                click.style(
                    "  ⊘ enrichment disabled (enrichment.ner_enabled/keyphrases_enabled=false)",
                    fg="yellow",
                )
            )
        else:
            click.echo(click.style("  ✓ enrichment done", fg="green"))
    else:
        click.echo(click.style("  ⊘ enrich skipped (using existing CleanText)", fg="yellow"))

    # Stage 3: Index (graph + shadow)
    if not skip_index:
        stage_banner("3/4 INDEX (CleanText + taxonomy + ontology → graph.pickle + shadow records)")
        # Build graph
        build_graph_from_clean_text(config=cfg, core_cfg=core_cfg, workers=w, overwrite=overwrite)
        click.echo(click.style("  ✓ graph done", fg="green"))
        # Build shadow records
        build_shadow_records(
            config=cfg, core_cfg=core_cfg, only_types=types, overwrite=overwrite, workers=w
        )
        click.echo(click.style("  ✓ shadow records done", fg="green"))
    else:
        click.echo(
            click.style(
                "  ⊘ index skipped (using existing graph.pickle + shadow/*.jsonl)", fg="yellow"
            )
        )
        require_graph(paths)
        require_shadow(paths, types)

    # Stage 4: Export
    if not skip_export:
        stage_banner("4/4 EXPORT (ShadowRecords → JSONL)")
        export_jsonl_per_type(config=cfg, only_types=types, overwrite=overwrite, workers=w)
        click.echo(click.style("  ✓ export done", fg="green"))
    else:
        click.echo(
            click.style("  ⊘ export skipped (using existing output/jsonl/*.jsonl)", fg="yellow")
        )
        require_exports(paths, types)

    click.echo(click.style("\n" + "═" * 60, fg="bright_green", bold=True))
    click.echo(
        click.style("  ✓ PIPELINE COMPLETED (deploy not included)", fg="bright_green", bold=True)
    )
    click.echo(click.style("═" * 60 + "\n", fg="bright_green", bold=True))


__all__ = ["cmd_run"]
