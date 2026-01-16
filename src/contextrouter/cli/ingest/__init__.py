"""Click CLI group for staged ingestion pipeline.

Important: this module must stay import-light. `contextrouter --help` should not
require importing ingestion providers/models and their optional dependencies.
"""

from __future__ import annotations

import importlib

import click

from contextrouter.cli.ingest.common import ALL_TYPES
from contextrouter.cli.registry import register_command

_INGEST_COMMANDS: dict[str, str] = {
    "preprocess": "contextrouter.cli.ingest.preprocess:cmd_preprocess",
    "structure": "contextrouter.cli.ingest.structure:cmd_structure",
    "enrich": "contextrouter.cli.ingest.enrich:cmd_enrich",
    "index": "contextrouter.cli.ingest.index:cmd_index",
    "persona": "contextrouter.cli.ingest.persona:cmd_persona",
    "deploy": "contextrouter.cli.ingest.deploy:cmd_deploy",
    "report": "contextrouter.cli.ingest.report:cmd_report",
    "run": "contextrouter.cli.ingest.run:cmd_run",
}

_INGEST_HELP: dict[str, str] = {
    "preprocess": "1. Prepare raw inputs (download/clean/split).",
    "persona": "2. Build persona artifacts.",
    "structure": "3. Build taxonomy and ontology artifacts.",
    "enrich": "3b. Enrich clean text with NER and keyphrases.",
    "index": "4. Build knowledge graph and shadow records.",
    "deploy": "5. Export, deploy to storage, and generate reports.",
    "report": "6. Generate ingestion reports (standalone).",
    "run": "Run the ingestion pipeline end-to-end (1-5).",
}


def _import_object(path: str) -> object:
    mod_name, attr = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


class _LazyIngestGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        # Return commands in pipeline order, not alphabetical
        return [
            "preprocess",
            "persona",
            "structure",
            "enrich",
            "index",
            "deploy",
            "report",
            "run",
        ]

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Avoid importing subcommands when the user runs `ingest --help`.
        rows: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            rows.append((name, _INGEST_HELP.get(name, "")))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = _INGEST_COMMANDS.get(cmd_name)
        if not target:
            return None
        obj = _import_object(target)
        return obj if isinstance(obj, click.Command) else None


@click.group(cls=_LazyIngestGroup)
def ingest() -> None:
    """Staged ingestion pipeline (lazy-loaded subcommands)."""


register_command("ingest", ingest)

__all__ = ["ingest", "ALL_TYPES"]
