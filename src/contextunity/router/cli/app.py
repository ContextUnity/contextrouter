"""Main Typer application root."""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Annotated

import typer
from contextunity.core import get_contextunit_logger, load_shared_config_from_env, setup_logging
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Trigger builtin command discovery (side-effect imports that call register_command).
from contextunity.router.cli import commands as _commands  # noqa: F401
from contextunity.router.cli.registry import iter_commands
from contextunity.router.core import Config, set_core_config
from contextunity.router.core.registry import scan

logger = get_contextunit_logger(__name__)

# Rich console for CLI error output
console = Console(stderr=True)
# Install rich traceback, but only when running CLI
install_rich_traceback(console=console, show_locals=False, suppress=[typer])

# Typer app definition
app = typer.Typer(
    name="contextrouter",
    help="ContextRouter CLI - LangGraph brain orchestrator and tools.",
    no_args_is_help=True,
    add_completion=True,
)


@app.callback()
def cli(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to settings.toml", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """ContextRouter CLI configuration."""
    # Setup logging from SharedConfig (with verbose override)
    config = load_shared_config_from_env()
    if verbose:
        from contextunity.core import LogLevel

        config.log_level = LogLevel.DEBUG

    # Use plain text format for CLI
    setup_logging(config=config, service_name="contextunity.router")

    # Keep CLI output readable
    if not verbose:
        try:
            from langchain_core._api.deprecation import LangChainDeprecationWarning
        except Exception:
            LangChainDeprecationWarning = Warning  # type: ignore[assignment]
        warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

    # Suppress verbose HTTP logging
    for logger_name in [
        "google",
        "google.genai",
        "google.auth",
        "google.api_core",
        "httpx",
        "httpcore",
        "urllib3",
    ]:
        get_contextunit_logger(logger_name).setLevel(logging.WARNING)

    # Load layered config
    cfg_obj = Config.load(str(config_path) if config_path else None)
    set_core_config(cfg_obj)

    # Plugin scanning
    for plugin_path in cfg_obj.plugins.paths or []:
        try:
            scan(Path(plugin_path))
        except Exception as e:
            logger.warning("Failed to scan plugin directory %s: %s", plugin_path, e)


# Register builtin commands
for name, cmd in iter_commands():
    if isinstance(cmd, typer.Typer):
        app.add_typer(cmd, name=name)
    else:
        # Since it's a TyperCommand, we can add it directly to Typer
        app.registered_commands.append(
            typer.models.CommandInfo(
                name=name,
                callback=cmd.callback,
                help=cmd.help,
            )
        )


def main() -> None:
    """Main entrypoint."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        # Default fallback, rich exception hook should catch it
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
