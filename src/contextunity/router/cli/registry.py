"""CLI command registry.

Command modules register Typer command groups at import time.
This keeps `cli/app.py` minimal and makes it easy to extend via plugins.
"""

from __future__ import annotations

from collections.abc import Iterator

import typer

_COMMANDS: dict[str, typer.Typer | typer.core.TyperCommand] = {}


def register_command(name: str, app: typer.Typer | typer.core.TyperCommand) -> None:
    _COMMANDS[name] = app


def iter_commands() -> Iterator[tuple[str, typer.Typer | typer.core.TyperCommand]]:
    yield from _COMMANDS.items()


__all__ = ["register_command", "iter_commands"]
