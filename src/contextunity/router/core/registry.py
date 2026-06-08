"""Simplified registry system with factory pattern.
Design goals:
- **Minimal abstraction** - only essential registries remain
- **Factory pattern** for core components (providers, connectors)
- **Direct imports** for static components where possible
- **Backward compatibility** - existing code continues to work
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk.interfaces import BaseTransformer

from contextunity.router.core.exceptions import RouterRegistryError
from contextunity.router.core.interfaces import BaseAgent, BaseConnector, BaseProvider
from contextunity.router.cortex.types import GraphFactoryProduct, RunnableGraphFactory

if TYPE_CHECKING:
    from contextunity.router.core.plugins import PluginContext

_VT = TypeVar("_VT")


@runtime_checkable
class ComponentFactoryFn(Protocol):
    """Dynamic plugin factory registered at import time (registry boundary)."""

    def __call__(self, /, **kwargs: object) -> object:
        """Construct a provider, connector, or transformer instance."""
        ...


def _import_component_type(module_name: str, class_name: str) -> type[object]:
    """Import a component class from a module path (dynamic registry boundary)."""
    module = importlib.import_module(module_name)
    raw: object = getattr(module, class_name, None)
    if raw is None or not isinstance(raw, type):
        raise RouterRegistryError(
            f"Missing or invalid class {class_name!r} in module {module_name!r}"
        )
    return raw


@runtime_checkable
class ZeroArgGraphThunk(Protocol):
    """Legacy graph registration: bare ``def builder(): ...`` with no ``build()`` method."""

    def __call__(self) -> GraphFactoryProduct:
        """Return compiled or uncompiled graph (registry resolution)."""
        ...


class _CallableGraphFactory:
    """Wrap a legacy zero-arg builder function as :class:`RunnableGraphFactory`."""

    __slots__: ClassVar[tuple[str, ...]] = ("_fn",)

    def __init__(self, fn: ZeroArgGraphThunk) -> None:
        self._fn: ZeroArgGraphThunk = fn

    def build(self) -> GraphFactoryProduct:
        return self._fn()


def as_runnable_graph_factory(
    builder: RunnableGraphFactory | ZeroArgGraphThunk,
) -> RunnableGraphFactory:
    """Normalize registry entries to :class:`RunnableGraphFactory`."""
    if isinstance(builder, RunnableGraphFactory):
        return builder
    if isinstance(builder, _CallableGraphFactory):
        return builder
    return _CallableGraphFactory(builder)


def _instantiate_transformer(
    cls: type[BaseTransformer],
    params: dict[str, object] | None = None,
) -> BaseTransformer:
    """Construct a transformer and apply ``configure(params)`` when provided."""
    instance = cls()
    if params:
        instance.configure(params)
    return instance


# ---- Graph Registry -------------------------------------------------


class Registry(Generic[_VT]):
    """Key-value store for dynamic component hot-swapping."""

    def __init__(self, *, name: str) -> None:
        """Create a registry with a human-readable *name*."""
        self._name: str = name
        self._items: dict[str, _VT] = {}

    def has(self, key: str) -> bool:
        """Return ``True`` if *key* is registered."""
        k = key.strip()
        return k in self._items

    def list_keys(self) -> list[str]:
        """List all available keys in the registry."""
        return sorted(self._items.keys())

    def get(self, key: str) -> _VT:
        """Look up *key*. Raises ``KeyError`` on miss."""
        k = key.strip()
        if k not in self._items:
            raise KeyError(f"{self._name}: unknown key '{k}'")
        return self._items[k]

    def register(self, key: str, value: _VT, *, overwrite: bool = False) -> None:
        """Add *value* under *key*. Raises ``KeyError`` if already registered and *overwrite* is ``False``."""
        k = key.strip()
        if not k:
            raise RouterRegistryError(f"{self._name}: registry key must be non-empty")
        if not overwrite and k in self._items:
            raise KeyError(f"{self._name}: '{k}' already registered")
        self._items[k] = value

    def unregister(self, key: str) -> bool:
        """Remove *key* from the registry; return ``True`` if it was present."""
        k = key.strip()
        return self._items.pop(k, None) is not None


# Initialize graph registry after Registry class is defined.
# All domain graphs are now YAML-template-driven
# and registered dynamically via RegisterManifest — no builtin_map needed.
graph_registry: Registry[RunnableGraphFactory] = Registry(name="graphs")


def register_graph(
    name: str,
) -> Callable[[RunnableGraphFactory | ZeroArgGraphThunk], RunnableGraphFactory]:
    """Decorator: register a graph factory under *name* in ``graph_registry``."""

    def decorator(
        factory: RunnableGraphFactory | ZeroArgGraphThunk,
    ) -> RunnableGraphFactory:
        """Register *factory* under *name* in the global graph registry."""
        normalized = as_runnable_graph_factory(factory)
        graph_registry.register(name, normalized)
        return normalized

    return decorator


# ---- Factory Classes ------------------------------------------------


class ComponentFactory:
    """Static factory resolving component names to instances via dynamic registries with built-in fallbacks."""

    # Dynamic factories populated by decorators
    _provider_factories: dict[str, ComponentFactoryFn] = {}
    _connector_factories: dict[str, ComponentFactoryFn] = {}
    _transformer_factories: dict[str, ComponentFactoryFn] = {}

    @staticmethod
    def create_provider(name: str, **kwargs: object) -> BaseProvider:
        """Instantiate a storage provider by *name*, checking dynamic factories first, then built-in fallbacks."""
        if name in ComponentFactory._provider_factories:
            created = ComponentFactory._provider_factories[name](**kwargs)
            if not isinstance(created, BaseProvider):
                raise RouterRegistryError(
                    f"Provider factory for '{name}' did not return BaseProvider"
                )
            return created

        # Fallback to built-in providers
        providers = {
            "brain": ("contextunity.router.modules.providers.storage.brain", "BrainProvider"),
        }

        if name not in providers:
            raise RouterRegistryError(f"Unknown provider: {name}")

        module_name, class_name = providers[name]
        cls_obj = _import_component_type(module_name, class_name)
        if not issubclass(cls_obj, BaseProvider):
            raise RouterRegistryError(f"Invalid provider class for '{name}'")
        provider_ctor: Callable[..., BaseProvider] = cls_obj
        return provider_ctor(**kwargs)

    @staticmethod
    def create_connector(name: str, **kwargs: object) -> BaseConnector:
        """Instantiate a data connector by *name*, checking dynamic factories first, then built-in fallbacks."""
        if name in ComponentFactory._connector_factories:
            created = ComponentFactory._connector_factories[name](**kwargs)
            if not isinstance(created, BaseConnector):
                raise RouterRegistryError(
                    f"Connector factory for '{name}' did not return BaseConnector"
                )
            return created

        # Fallback to built-in connectors
        connectors = {
            "web": ("contextunity.router.modules.connectors.web", "WebSearchConnector"),
            "web_scraper": (
                "contextunity.router.modules.connectors.web",
                "WebScraperConnector",
            ),
            "file": ("contextunity.router.modules.connectors.file", "FileConnector"),
            "rss": ("contextunity.router.modules.connectors.rss", "RSSConnector"),
            "api": ("contextunity.router.modules.connectors.api", "APIConnector"),
        }

        if name not in connectors:
            raise RouterRegistryError(f"Unknown connector: {name}")

        module_name, class_name = connectors[name]
        cls_obj = _import_component_type(module_name, class_name)
        if not issubclass(cls_obj, BaseConnector):
            raise RouterRegistryError(f"Invalid connector class for '{name}'")
        connector_ctor: Callable[..., BaseConnector] = cls_obj
        return connector_ctor(**kwargs)

    @staticmethod
    def create_transformer(name: str, **kwargs: object) -> BaseTransformer:
        """Instantiate a transformer by *name*, checking dynamic factories first, then built-in fallbacks."""
        params: dict[str, object] | None = dict(kwargs) if kwargs else None
        if name in ComponentFactory._transformer_factories:
            created = ComponentFactory._transformer_factories[name](**kwargs)
            if not isinstance(created, BaseTransformer):
                raise RouterRegistryError(
                    f"Transformer factory for '{name}' did not return BaseTransformer"
                )
            return created

        # Fallback to built-in transformers
        transformers = {
            "metadata_mapper": (
                "contextunity.router.modules.transformers.metadata",
                "MetadataMapper",
            ),
            "summarizer": (
                "contextunity.router.modules.transformers.summarization",
                "Summarizer",
            ),
        }

        if name not in transformers:
            raise RouterRegistryError(f"Unknown transformer: {name}")

        module_name, class_name = transformers[name]
        cls_obj = _import_component_type(module_name, class_name)
        if not issubclass(cls_obj, BaseTransformer):
            raise RouterRegistryError(f"Invalid transformer class for '{name}'")
        transformer_cls: type[BaseTransformer] = cls_obj
        return _instantiate_transformer(transformer_cls, params)

    @classmethod
    def bind_connector(cls, name: str, connector_cls: type[BaseConnector]) -> None:
        """Register a connector class for dynamic resolution."""
        _connector_registry[name] = connector_cls

        def _factory(**kwargs: object) -> BaseConnector:
            return connector_cls(**kwargs)

        cls._connector_factories[name] = _factory

    @classmethod
    def bind_provider(cls, name: str, provider_cls: type[BaseProvider]) -> None:
        """Register a provider class for dynamic resolution."""
        _provider_registry[name] = provider_cls

        def _factory(**kwargs: object) -> BaseProvider:
            return provider_cls(**kwargs)

        cls._provider_factories[name] = _factory

    @classmethod
    def bind_transformer(
        cls,
        name: str,
        transformer_cls: type[BaseTransformer],
    ) -> None:
        """Register a transformer class for dynamic resolution."""

        def _factory(**kw: object) -> BaseTransformer:
            params: dict[str, object] | None = dict(kw) if kw else None
            return _instantiate_transformer(transformer_cls, params)

        _transformer_registry[name] = transformer_cls
        cls._transformer_factories[name] = _factory


# ---- Component Registration (Dynamic Registries) ----

# Dynamic registries for hot-swapping components
_provider_registry: dict[str, type[BaseProvider]] = {}
_connector_registry: dict[str, type[BaseConnector]] = {}
_transformer_registry: dict[str, type[BaseTransformer]] = {}


def register_agent(name: str) -> Callable[[type[BaseAgent]], type[BaseAgent]]:
    """Decorator: register an agent class under *name* in ``agent_registry``."""

    def decorator(cls: type[BaseAgent]) -> type[BaseAgent]:
        """Register *cls* and return it unchanged."""
        agent_registry.register(name, cls, overwrite=True)
        return cls

    return decorator


def register_connector(name: str) -> Callable[[type[BaseConnector]], type[BaseConnector]]:
    """Decorator: register a connector class under *name* (accessible via ``select_connector``)."""

    def decorator(cls: type[BaseConnector]) -> type[BaseConnector]:
        """Register *cls* and return it unchanged."""
        ComponentFactory.bind_connector(name, cls)
        return cls

    return decorator


def register_provider(name: str) -> Callable[[type[BaseProvider]], type[BaseProvider]]:
    """Decorator: register a storage provider class under *name* (accessible via ``select_provider``)."""

    def decorator(cls: type[BaseProvider]) -> type[BaseProvider]:
        """Register *cls* and return it unchanged."""
        ComponentFactory.bind_provider(name, cls)
        return cls

    return decorator


def register_transformer(name: str) -> Callable[[type[BaseTransformer]], type[BaseTransformer]]:
    """Decorator: register a transformer class under *name* (accessible via ``select_transformer``)."""

    def decorator(cls: type[BaseTransformer]) -> type[BaseTransformer]:
        """Register *cls* and return it unchanged."""
        ComponentFactory.bind_transformer(name, cls)
        return cls

    return decorator


# ---- Dynamic Selection Functions ----


def select_provider(name: str, **kwargs: object) -> BaseProvider:
    """Instantiate a provider: check the dynamic registry first, fall back to ``ComponentFactory``."""
    if name in _provider_registry:
        return _provider_registry[name](**kwargs)
    return ComponentFactory.create_provider(name, **kwargs)


def select_connector(name: str, **kwargs: object) -> BaseConnector:
    """Instantiate a connector: check the dynamic registry first, fall back to ``ComponentFactory``."""
    if name in _connector_registry:
        return _connector_registry[name](**kwargs)
    return ComponentFactory.create_connector(name, **kwargs)


def select_transformer(name: str, **kwargs: object) -> BaseTransformer:
    """Instantiate a transformer: check the dynamic registry first, fall back to ``ComponentFactory``."""
    params: dict[str, object] | None = dict(kwargs) if kwargs else None
    if name in _transformer_registry:
        return _instantiate_transformer(_transformer_registry[name], params)
    return ComponentFactory.create_transformer(name, **kwargs)


# ---- Agent Registry ----
# Essential for cortex agent hot-swapping and dynamic graph assembly

agent_registry: Registry[type[BaseAgent]] = Registry(name="agents")

# ---- Plugin scanning -------------------------------------------------------

logger = get_contextunit_logger(__name__)


def scan(plugin_dir: Path) -> list[PluginContext]:
    """Scan a directory for manifest-based plugins.

    Each plugin is a subdirectory containing a ``plugin.yaml`` manifest.
    Plugins are loaded via :func:`contextunity.router.core.plugins.load_plugin`
    and receive a capability-gated :class:`PluginContext`.

    Args:
        plugin_dir: Directory to scan for plugin subdirectories.

    Returns:
        List of loaded PluginContext instances.
    """
    if not plugin_dir.exists() or not plugin_dir.is_dir():
        logger.debug("Plugin directory does not exist: %s", plugin_dir)
        return []

    loaded: list[PluginContext] = []

    for subdir in sorted(plugin_dir.iterdir()):
        if not subdir.is_dir():
            continue
        manifest_file = subdir / "plugin.yaml"
        if not manifest_file.exists():
            manifest_file = subdir / "plugin.yml"
        if not manifest_file.exists():
            continue

        try:
            from contextunity.router.core.plugins import load_plugin

            ctx = load_plugin(subdir)
            if ctx is not None:
                loaded.append(ctx)
        except Exception as e:
            logger.error("Failed to load plugin from %s: %s", subdir, e)

    return loaded


__all__ = [
    "ComponentFactory",
    "agent_registry",  # Essential for cortex agent hot-swapping
    "ZeroArgGraphThunk",
    "as_runnable_graph_factory",
    "graph_registry",  # Dynamic graph registration
    "register_agent",  # For custom agents
    "register_connector",  # For custom connectors
    "register_graph",  # For inline/project graphs
    "register_provider",  # For custom providers
    "register_transformer",  # For custom transformers
    "select_provider",  # Dynamic provider selection
    "select_connector",  # Dynamic connector selection
    "scan",  # Plugin directory scanning
    "select_transformer",  # Dynamic transformer selection
]
