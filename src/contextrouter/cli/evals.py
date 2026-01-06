"""RAG evaluation commands."""

from __future__ import annotations

from pathlib import Path

import click

from contextrouter.cli.registry import register_command


@click.group("eval")
def evals() -> None:
    """Run automated evaluations on RAG responses."""


register_command("eval", evals)


@evals.command("run")
@click.argument("test-set", type=click.Path(exists=True, path_type=Path))
def cmd_run_eval(test_set: Path) -> None:
    """Run evaluation on a test set (JSON)."""
    click.echo(f"Running evaluation on {test_set}...")
    click.echo("Status: Stubbed. Implement an Evaluator in cortex.evals.base to continue.")
