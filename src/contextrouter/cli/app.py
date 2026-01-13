"""Main Click application root."""

from __future__ import annotations

import logging
from pathlib import Path

import click

# Trigger builtin command discovery (side-effect imports that call register_command).
from contextrouter.cli import commands as _commands  # noqa: F401
from contextrouter.cli.registry import iter_commands
from contextrouter.core.config import get_core_config, set_core_config
from contextrouter.core.config.main import Config
from contextrouter.core.registry import scan

logger = logging.getLogger(__name__)


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
