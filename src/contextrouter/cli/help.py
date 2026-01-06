"""Top-level CLI help command.

This command exists to provide a stable UX entrypoint for users who don't know
which subcommand to run yet.
"""

from __future__ import annotations

import textwrap

import click

from contextrouter.cli.registry import iter_commands, register_command


@click.command("help")
def cmd_help() -> None:
    """Show an overview of available command groups."""
    rows = list(iter_commands())
    if not rows:
        click.echo("No commands registered.")
        return

    click.echo("ContextRouter CLI")
    click.echo("")
    click.echo("Available command groups:")

    for name, cmd in sorted(rows, key=lambda x: x[0]):
        # Click keeps help text in .help / .short_help depending on type.
        short = getattr(cmd, "short_help", None) or getattr(cmd, "help", None) or ""
        short = " ".join(str(short).split())
        click.echo(f"  - {name:10s} {short}")

    click.echo("")
    click.echo("Tips:")
    click.echo(
        textwrap.dedent("""\
      - Run `contextrouter <group> --help` for group-specific commands.
      - Golden path: `contextrouter rag query "..."` (Vertex AI Search + Gemini on Vertex).
    """).rstrip()
    )


register_command("help", cmd_help)


__all__ = ["cmd_help"]
