"""Simplified registry system with factory pattern.

Design goals:
- **Minimal abstraction** - only essential registries remain
- **Factory pattern** for core components (providers, connectors)
- **Direct imports** for static components where possible
- **Backward compatibility** - existing code continues to work
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

# ---- Graph Registry -------------------------------------------------

# Graph registry defined later in file, forward reference for now
graph_registry: "Registry"


def register_graph(name: str) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """Decorator to register a custom graph builder.

    Args:
        name: Graph name/key for lookup in config

    Returns:
        Decorator function

    Example:
        @register_graph("my_custom_graph")
        def build_my_graph():
            # Custom graph building logic
            return StateGraph(...)
    """

    def decorator(func: Callable[[], object]) -> Callable[[], object]:
        graph_registry.register(name, func)
        return func

    return decorator


# ---- Factory Classes ------------------------------------------------


class ComponentFactory:
    """Factory for creating core components."""

    # Dynamic factories populated by decorators
    _provider_factories: dict[str, Any] = {}
    _connector_factories: dict[str, Any] = {}
    _transformer_factories: dict[str, Any] = {}

    @staticmethod
    def create_provider(name: str, **kwargs: Any) -> Any:
        """Create a storage provider instance."""
        if name in ComponentFactory._provider_factories:
            return ComponentFactory._provider_factories[name](**kwargs)

        # Fallback to built-in providers
        providers = {
            "vertex": (
                "contextrouter.modules.providers.storage.vertex",
                "VertexProvider",
            ),
            "postgres": (
                "contextrouter.modules.providers.storage.postgres.provider",
                "PostgresProvider",
            ),
            "gcs": ("contextrouter.modules.providers.storage.gcs", "GCSProvider"),
        }

        if name not in providers:
            raise ValueError(f"Unknown provider: {name}")

        module_name, class_name = providers[name]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def create_connector(name: str, **kwargs: Any) -> Any:
        """Create a data connector instance."""
        if name in ComponentFactory._connector_factories:
            return ComponentFactory._connector_factories[name](**kwargs)

        # Fallback to built-in connectors
        connectors = {
            "web": ("contextrouter.modules.connectors.web", "WebSearchConnector"),
            "web_scraper": (
                "contextrouter.modules.connectors.web",
                "WebScraperConnector",
            ),
            "file": ("contextrouter.modules.connectors.file", "FileConnector"),
            "rss": ("contextrouter.modules.connectors.rss", "RSSConnector"),
            "api": ("contextrouter.modules.connectors.api", "APIConnector"),
        }

        if name not in connectors:
            raise ValueError(f"Unknown connector: {name}")

        module_name, class_name = connectors[name]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def create_transformer(name: str, **kwargs: Any) -> Any:
        """Create a transformer instance."""
        if name in ComponentFactory._transformer_factories:
            return ComponentFactory._transformer_factories[name](**kwargs)

        # Fallback to built-in transformers
        transformers = {
            "metadata_mapper": (
                "contextrouter.modules.transformers.metadata",
                "MetadataMapper",
            ),
            "summarizer": (
                "contextrouter.modules.transformers.summarization",
                "Summarizer",
            ),
        }

        if name not in transformers:
            raise ValueError(f"Unknown transformer: {name}")

        module_name, class_name = transformers[name]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)


def _lazy_import_object(path: str) -> object:
    """Import an object by dotted path."""
    raw = (path or "").strip()
    if not raw:
        raise ValueError("Empty import path")
    if ":" in raw:
        mod_name, attr = raw.split(":", 1)
    elif "." in raw:
        mod_name, attr = raw.rsplit(".", 1)
    else:
        mod_name = raw
        attr = raw
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


class Registry:
    """Minimal registry for dynamic component registration."""

    def __init__(self, *, name: str, builtin_map: dict[str, str] | None = None) -> None:
        self._name = name
        self._items: dict[str, Any] = {}
        self._builtin_map: dict[str, str] = builtin_map or {}

    def has(self, key: str) -> bool:
        """Check if a key exists in the registry."""
        k = key.strip()
        return k in self._items or k in self._builtin_map

    def list_keys(self) -> list[str]:
        """List all available keys in the registry."""
        return sorted(set(self._items.keys()) | set(self._builtin_map.keys()))

    def get(self, key: str) -> Any:
        k = key.strip()
        if k not in self._items and k in self._builtin_map:
            self._items[k] = _lazy_import_object(self._builtin_map[k])
        if k not in self._items:
            raise KeyError(f"{self._name}: unknown key '{k}'")
        return self._items[k]

    def register(self, key: str, value: Any, *, overwrite: bool = False) -> None:
        k = key.strip()
        if not k:
            raise ValueError(f"{self._name}: registry key must be non-empty")
        if not overwrite and k in self._items:
            raise KeyError(f"{self._name}: '{k}' already registered")
        self._items[k] = value


# Initialize graph registry after Registry class is defined
graph_registry = Registry(name="graphs", builtin_map={})

# ---- Component Registration (Dynamic Registries) ----

# Dynamic registries for hot-swapping components
_provider_registry: dict[str, Any] = {}
_connector_registry: dict[str, Any] = {}
_transformer_registry: dict[str, Any] = {}


def register_agent(name: str) -> Any:
    """Register an agent class."""

    def decorator(cls: Any) -> Any:
        agent_registry.register(name, cls, overwrite=True)
        return cls

    return decorator


def register_connector(name: str) -> Any:
    """Register a connector class for dynamic selection."""

    def decorator(cls: Any) -> Any:
        _connector_registry[name] = cls
        ComponentFactory._connector_factories[name] = lambda **kwargs: cls(**kwargs)
        return cls

    return decorator


def register_provider(name: str) -> Any:
    """Register a provider class for dynamic selection."""

    def decorator(cls: Any) -> Any:
        _provider_registry[name] = cls
        ComponentFactory._provider_factories[name] = lambda **kwargs: cls(**kwargs)
        return cls

    return decorator


def register_transformer(name: str) -> Any:
    """Register a transformer class for dynamic selection."""

    def decorator(cls: Any) -> Any:
        _transformer_registry[name] = cls
        ComponentFactory._transformer_factories[name] = lambda **kwargs: cls(**kwargs)
        return cls

    return decorator


# ---- Dynamic Selection Functions ----


def select_provider(name: str, **kwargs: Any) -> Any:
    """Dynamically select a provider from registry."""
    if name in _provider_registry:
        return _provider_registry[name](**kwargs)
    return ComponentFactory.create_provider(name, **kwargs)


def select_connector(name: str, **kwargs: Any) -> Any:
    """Dynamically select a connector from registry."""
    if name in _connector_registry:
        return _connector_registry[name](**kwargs)
    return ComponentFactory.create_connector(name, **kwargs)


def select_transformer(name: str, **kwargs: Any) -> Any:
    """Dynamically select a transformer from registry."""
    if name in _transformer_registry:
        return _transformer_registry[name](**kwargs)
    return ComponentFactory.create_transformer(name, **kwargs)


# ---- Agent Registry ----
# Essential for cortex agent hot-swapping and dynamic graph assembly

BUILTIN_AGENTS: dict[str, str] = {
    "extract_query": "contextrouter.cortex.nodes.rag_retrieval.extract.ExtractQueryAgent",
    "detect_intent": "contextrouter.cortex.nodes.rag_retrieval.intent.DetectIntentAgent",
    "retrieve": "contextrouter.cortex.nodes.rag_retrieval.retrieve.RetrieveAgent",
    "suggest": "contextrouter.cortex.nodes.rag_retrieval.suggest.SuggestAgent",
    "generate": "contextrouter.cortex.nodes.rag_retrieval.generate.GenerateAgent",
    "routing": "contextrouter.cortex.nodes.rag_retrieval.routing.RoutingAgent",
}

agent_registry: Registry = Registry(name="agents", builtin_map=BUILTIN_AGENTS)


# ---- Plugin scanning -------------------------------------------------------

logger = logging.getLogger(__name__)


def scan(plugin_dir: Path) -> None:
    """Scan a directory for Python plugins and import them.

    This allows users to extend ContextRouter with custom components
    (agents, tools, connectors, etc.) without modifying the core code.

    Args:
        plugin_dir: Directory containing Python plugin modules
    """
    if not plugin_dir.exists() or not plugin_dir.is_dir():
        logger.debug(f"Plugin directory does not exist: {plugin_dir}")
        return

    # Find all .py files in the directory
    plugin_files = list(plugin_dir.glob("*.py"))
    if not plugin_files:
        logger.debug(f"No Python files found in plugin directory: {plugin_dir}")
        return

    logger.info(f"Scanning {len(plugin_files)} plugin files in {plugin_dir}")

    # Import each plugin file
    for plugin_file in plugin_files:
        if plugin_file.name.startswith("_"):
            continue  # Skip private files

        try:
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                logger.info(f"Loaded plugin: {module_name} from {plugin_file}")
            else:
                logger.warning(f"Could not load plugin: {plugin_file}")
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_file}: {e}")


__all__ = [
    "ComponentFactory",
    "agent_registry",  # Essential for cortex agent hot-swapping
    "graph_registry",  # Dynamic graph registration
    "register_agent",  # For custom agents
    "register_connector",  # For custom connectors
    "register_graph",  # For custom graphs
    "register_provider",  # For custom providers
    "register_transformer",  # For custom transformers
    "select_provider",  # Dynamic provider selection
    "select_connector",  # Dynamic connector selection
    "scan",  # Plugin directory scanning
    "select_transformer",  # Dynamic transformer selection
]
