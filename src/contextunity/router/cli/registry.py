"""CLI command registry.
Command modules register Typer command groups at import time.
This keeps `cli/app.py` minimal and makes it easy to extend via plugins.
"""

from __future__ import annotations

from collections.abc import Iterator

import typer
from typer.core import TyperCommand

_COMMANDS: dict[str, typer.Typer | TyperCommand] = {}


def register_command(name: str, app: typer.Typer | TyperCommand) -> None:
    """Add *app* to the global command map under *name* for later mounting by the CLI root."""
    _COMMANDS[name] = app


def iter_commands() -> Iterator[tuple[str, typer.Typer | TyperCommand]]:
    """Yield ``(name, app)`` pairs for all registered CLI command groups."""
    yield from _COMMANDS.items()


__all__ = ["register_command", "iter_commands"]
