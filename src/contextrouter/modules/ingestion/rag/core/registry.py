"""Plugin registry for ingestion plugins."""

from __future__ import annotations

from typing import Type

from .plugins import IngestionPlugin

_PLUGINS: dict[str, Type[IngestionPlugin]] = {}


def register_plugin(source_type: str):
    """Decorator to register a plugin class.

    Args:
        source_type: The source type string (e.g., "video", "book")

    Example:
        @register_plugin("video")
        class VideoPlugin(IngestionPlugin):
            ...
    """

    def decorator(cls: Type[IngestionPlugin]) -> Type[IngestionPlugin]:
        _PLUGINS[source_type] = cls
        return cls

    return decorator


def get_plugin_class(source_type: str) -> Type[IngestionPlugin]:
    """Retrieve a plugin class by source type.

    Args:
        source_type: The source type string

    Returns:
        The plugin class

    Raises:
        ValueError: If no plugin is registered for the given source type
    """
    if source_type not in _PLUGINS:
        raise ValueError(f"No plugin registered for: {source_type}")
    return _PLUGINS[source_type]


def get_all_plugins() -> list[Type[IngestionPlugin]]:
    """Get all registered plugins.

    Returns:
        List of all registered plugin classes
    """
    return list(_PLUGINS.values())
