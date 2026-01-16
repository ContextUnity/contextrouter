"""Main Click application root with Typer-enhanced error handling."""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import click
from rich.console import Console
from rich.traceback import Traceback

# Trigger builtin command discovery (side-effect imports that call register_command).
from contextrouter.cli import commands as _commands  # noqa: F401
from contextrouter.cli.registry import iter_commands
from contextrouter.core import Config, get_core_config, set_core_config
from contextrouter.core.registry import scan

logger = logging.getLogger(__name__)

# Rich console for CLI error output only (not installed globally to avoid affecting runtime traces)
console = Console(stderr=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to settings.toml",
)
@click.pass_context
def cli(ctx, verbose, config_path):
    """Contextrouter CLI - LangGraph brain orchestrator and tools."""
    ctx.ensure_object(dict)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    # Keep CLI output readable: suppress noisy LangChain deprecation warnings by default.
    # This is CLI-only; library/runtime users still get full warnings by default.
    if not verbose:
        try:
            from langchain_core._api.deprecation import LangChainDeprecationWarning
        except Exception:
            LangChainDeprecationWarning = Warning  # type: ignore[assignment]
        warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

    # Suppress verbose HTTP logging from Google API clients
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Load layered config for CLI session (standalone-friendly).
    # - Defaults < env < TOML < overrides
    # - `.env` is optional and auto-detected in the working directory.
    set_core_config(Config.load(config_path))

    # Plugin scanning - load user extensions
    cfg = get_core_config()
    for plugin_path in cfg.plugins.paths or []:
        try:
            scan(Path(plugin_path))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to scan plugin directory {plugin_path}: {e}")


# Register builtin commands (side-effect imports)

for name, cmd in iter_commands():
    cli.add_command(cmd, name=name)


def _handle_exception(exc: Exception) -> None:
    """Enhanced error handler using Rich for beautiful error output (CLI-only, doesn't affect runtime traces)."""
    if isinstance(exc, click.ClickException):
        # Click exceptions already have formatted messages
        console.print(f"[bold red]Error:[/bold red] {exc.format_message()}")
        sys.exit(exc.exit_code)
    elif isinstance(exc, click.Abort):
        console.print("[yellow]Aborted by user[/yellow]")
        sys.exit(1)
    elif isinstance(exc, KeyboardInterrupt):
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    else:
        # Use Rich traceback for unhandled exceptions (CLI context only)
        # This doesn't affect runtime traces in web servers or other parts of the app
        traceback = Traceback.from_exception(
            type(exc),
            exc,
            exc.__traceback__,
            show_locals=False,
            max_frames=20,
            suppress=[click],
        )
        console.print(traceback)
        sys.exit(1)


def main() -> None:
    """Main entrypoint with enhanced error handling."""
    try:
        cli(standalone_mode=False)
    except Exception as exc:
        _handle_exception(exc)
