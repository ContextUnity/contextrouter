"""Configuration and path constants for the ingestion pipeline.

This module provides:
- Path constants for assets folder structure
- TOML configuration loading
- Default values for all settings

Environment Variables:
- CONTEXTROUTER_CONFIG_PATH: Direct path to settings.toml config file
- CONTEXTROUTER_ASSETS_PATH: Path to assets folder (config derived as {assets}/settings.toml)

Priority: CONTEXTROUTER_CONFIG_PATH > CONTEXTROUTER_ASSETS_PATH > package default
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from contextrouter.core import get_env

from .settings import RagIngestionConfig

logger = logging.getLogger(__name__)

# Environment variable names
ENV_CONFIG_PATH = "CONTEXTROUTER_CONFIG_PATH"
ENV_ASSETS_PATH = "CONTEXTROUTER_ASSETS_PATH"

_THIS_FILE = Path(__file__).resolve()
_INGESTION_DIR = _THIS_FILE.parent  # ingestion/rag/

# Typed defaults live in RagIngestionConfig. Keep these path defaults derived from it.
_DEFAULT_CFG = RagIngestionConfig()
DEFAULT_CONFIG_PATH = _DEFAULT_CFG.paths.assets_folder / "settings.toml"
DEFAULT_TAXONOMY_PATH = _DEFAULT_CFG.assets_paths()["taxonomy"]
DEFAULT_GRAPH_PATH = _DEFAULT_CFG.assets_paths()["graph"]
DEFAULT_ONTOLOGY_PATH = _DEFAULT_CFG.assets_paths()["ontology"]


_cached_config: RagIngestionConfig | None = None
_cached_config_path: Path | None = None


def _resolve_config_path() -> Path:
    """Resolve config path from environment or defaults.

    Priority:
    1. CONTEXTROUTER_CONFIG_PATH env var (direct path to config file)
    2. CONTEXTROUTER_ASSETS_PATH env var (derives config as {assets}/settings.toml)
    3. Default: ./assets/ingestion/settings.toml (relative to current working directory)
    3. Default: ./assets/ingestion/settings.toml (relative to current working directory)
    """
    # 1. Direct config path override
    if env_config := get_env(ENV_CONFIG_PATH):
        return Path(env_config).resolve()

    # 2. Assets path override (config is {assets}/settings.toml)
    if env_assets := get_env(ENV_ASSETS_PATH):
        return Path(env_assets).resolve() / "settings.toml"

    # 3. Package default (from typed defaults)
    return DEFAULT_CONFIG_PATH


def _log_config_not_found(config_path: Path, example_path: Path) -> None:
    """Log helpful message when config file is not found."""
    msg_lines = [
        f"Config file not found: {config_path}",
        "",
        "To configure the ingestion pipeline, set the assets path and copy the example config:",
        f"  export {ENV_ASSETS_PATH}=/path/to/your/assets/folder",
        f"  cp {example_path} $CONTEXTROUTER_ASSETS_PATH/settings.toml",
        "",
        "Using default configuration.",
    ]
    logger.warning("\n".join(msg_lines))


def load_config(config_path: Path | None = None, *, force: bool = False) -> RagIngestionConfig:
    """Load configuration from TOML file (cached).

    Args:
        config_path: Path to TOML config file. If None, resolves from env vars or defaults.
        force: If True, bypass cache and reload.

    Returns:
        Configuration dictionary with defaults applied for missing values.

    Environment Variables:
        CONTEXTROUTER_CONFIG_PATH: Direct path to config file (highest priority)
        CONTEXTROUTER_ASSETS_PATH: Path to assets folder (config = {assets}/settings.toml)
    """
    global _cached_config, _cached_config_path

    resolved_path = config_path or _resolve_config_path()

    # Return cached config if same path and not forcing reload
    if not force and _cached_config is not None and _cached_config_path == resolved_path:
        return _cached_config

    if not resolved_path.exists():
        # Provide helpful guidance based on what's missing
        example_path = _INGESTION_DIR / "settings.toml.example"
        _log_config_not_found(resolved_path, example_path)
        cfg = RagIngestionConfig()
        if env_assets := get_env(ENV_ASSETS_PATH):
            cfg.paths.assets_folder = Path(env_assets).resolve()
        _cached_config = cfg
        _cached_config_path = resolved_path
        return _cached_config

    config_path = resolved_path

    try:
        import tomllib
    except ImportError:
        # Python < 3.11 fallback
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "Neither tomllib (Python 3.11+) nor tomli available. Using default configuration."
            )
            cfg = RagIngestionConfig()
            if env_assets := get_env(ENV_ASSETS_PATH):
                cfg.paths.assets_folder = Path(env_assets).resolve()
            _cached_config = cfg
            _cached_config_path = config_path
            return _cached_config

    try:
        with open(config_path, "rb") as f:
            file_config = tomllib.load(f)
        cfg = RagIngestionConfig.model_validate(file_config)
        if env_assets := get_env(ENV_ASSETS_PATH):
            cfg.paths.assets_folder = Path(env_assets).resolve()
        _cached_config = cfg
        _cached_config_path = config_path
        logger.info("Loaded config from %s", config_path)
        return cfg
    except Exception as e:
        logger.warning("Failed to load config from %s: %s. Using defaults.", config_path, e)
        cfg = RagIngestionConfig()
        if env_assets := get_env(ENV_ASSETS_PATH):
            cfg.paths.assets_folder = Path(env_assets).resolve()
        _cached_config = cfg
        _cached_config_path = config_path
        return cfg


def get_assets_paths(config: RagIngestionConfig | None = None) -> dict[str, Path]:
    """Get standardized asset paths from configuration.

    Priority for assets_folder:
    1. CONTEXTROUTER_ASSETS_PATH env var
    2. Config file paths.assets_folder setting
    3. Package default

    Args:
        config: Configuration dictionary. If None, uses defaults.

    Returns:
        Dictionary with Path objects for each asset directory.
    """
    cfg = config or load_config()
    return cfg.assets_paths()


def get_plugin_source_dir(
    plugin_type: str,
    config: RagIngestionConfig | None = None,
    plugin_instance: Any | None = None,
) -> Path:
    """Get source directory for a specific plugin type.

    Priority:
    1. Config override: [plugins.{plugin_type}].dir
    2. Plugin default: plugin.default_source_dir
    3. Fallback: plugin_type

    Args:
        plugin_type: Plugin source type (video, book, qa, web, knowledge)
        config: Configuration dictionary. If None, uses defaults.
        plugin_instance: Optional plugin instance to get default_source_dir

    Returns:
        Path to plugin's source directory
    """
    cfg = config or load_config()
    paths = get_assets_paths(cfg)

    plugin_dir: str | None = getattr(getattr(cfg.plugins, plugin_type, None), "dir", None)
    if not plugin_dir and plugin_instance and hasattr(plugin_instance, "default_source_dir"):
        plugin_dir = str(plugin_instance.default_source_dir)
    plugin_dir = plugin_dir or plugin_type
    return paths["source"] / plugin_dir


def ensure_directories_exist(paths: dict[str, Path] | None = None) -> None:
    """Ensure all required asset directories exist.

    Args:
        paths: Dictionary of paths from get_assets_paths(). If None, uses defaults.
    """
    if paths is None:
        paths = get_assets_paths()

    for key in ["source", "clean_text", "shadow", "upload", "jsonl", "processing"]:
        path = paths.get(key)
        if path and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", path)
