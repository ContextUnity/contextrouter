"""CLI command registry.

Command modules register Click command groups at import time.
This keeps `cli/app.py` minimal and makes it easy to extend via plugins.
"""

from __future__ import annotations

from collections.abc import Iterator

import click

_COMMANDS: dict[str, click.BaseCommand] = {}


def register_command(name: str, cmd: click.BaseCommand) -> None:
    _COMMANDS[name] = cmd


def iter_commands() -> Iterator[tuple[str, click.BaseCommand]]:
    yield from _COMMANDS.items()


__all__ = ["register_command", "iter_commands"]
