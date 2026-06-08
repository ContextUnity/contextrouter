"""Plugin manifest and context for contextunity.router plugin system.
Plugins are directories containing:
- plugin.yaml — manifest with metadata, capabilities, and requirements
- entry_point.py — Python module with on_load(ctx: PluginContext) function
Plugins without a manifest are loaded in legacy mode (bare .py files).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk.interfaces import BaseTransformer
from contextunity.core.types import JsonDict, is_json_dict
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from contextunity.router.core.exceptions import RouterPluginError
from contextunity.router.core.interfaces import BaseConnector, BaseProvider
from contextunity.router.core.registry import ZeroArgGraphThunk
from contextunity.router.cortex.types import RunnableGraphFactory

logger = get_contextunit_logger(__name__)

# ============================================================================
# Plugin Manifest (parsed from plugin.yaml)
# ============================================================================


class PluginCapability(str, Enum):
    """Capabilities a plugin can request."""

    TOOLS = "tools"
    GRAPHS = "graphs"
    CONNECTORS = "connectors"
    PROVIDERS = "providers"
    TRANSFORMERS = "transformers"


class PluginManifest(BaseModel):
    """Plugin manifest parsed from plugin.yaml.

    Example plugin.yaml:

        name: my-enrichment-plugin
        version: 1.0.0
        description: Custom enrichment tools for product data
        author: Acme Corp
        requires:
          contextunity.router: ">=0.9.0"
        capabilities:
          - tools
          - graphs
        entry_point: plugin.py
    """

    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+")
    description: str = Field(default="")
    author: str = Field(default="")
    capabilities: list[PluginCapability] = Field(default_factory=list)
    entry_point: str = Field(default="plugin.py")
    requires: dict[str, str] = Field(default_factory=dict)
    enabled: bool = Field(default=True)

    @field_validator("entry_point")
    @classmethod
    def validate_entry_point(cls, v: str) -> str:
        """Reject entry points with path traversal or non-``.py`` extensions."""
        if not v.endswith(".py"):
            raise ValueError("entry_point must be a .py file")
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("entry_point must not contain path separators")
        return v


# ============================================================================
# Plugin Context (capability-mediated access)
# ============================================================================


class PluginContext:
    """Mediated access to contextunity.router.services.

    Plugins receive a PluginContext that restricts access to only the
    capabilities declared in their manifest.

    Usage in plugin.py:

        def on_load(ctx: PluginContext) -> None:
            from langchain_core.tools import tool

            @tool
            def my_tool(query: str) -> str:
                '''My custom tool.'''
                return f"Result for {query}"

            ctx.register_tool(my_tool)
    """

    def __init__(self, manifest: PluginManifest, *, plugin_dir: Path) -> None:
        """Store *manifest*, derive the capability set, and initialise per-type registration trackers."""
        self._manifest: PluginManifest = manifest
        self._capabilities: set[PluginCapability] = set(manifest.capabilities)
        self._plugin_dir: Path = plugin_dir
        self._registered_tools: list[str] = []
        self._registered_graphs: list[str] = []
        self._registered_connectors: list[str] = []
        self._registered_providers: list[str] = []
        self._registered_transformers: list[str] = []

    # ---- Properties --------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable plugin identifier from the manifest."""
        return self._manifest.name

    @property
    def version(self) -> str:
        """Semver string declared in the plugin manifest."""
        return self._manifest.version

    @property
    def capabilities(self) -> set[PluginCapability]:
        """Copy of the capability flags granted to this plugin at load time."""
        return self._capabilities.copy()

    @property
    def plugin_dir(self) -> Path:
        """Absolute path to the plugin’s directory on disk."""
        return self._plugin_dir

    # ---- Capability-gated registration -------------------------------------

    def _check_capability(self, cap: PluginCapability) -> None:
        """Raise securityerror if capability not granted."""
        if cap not in self._capabilities:
            from contextunity.core.exceptions import SecurityError

            raise SecurityError(
                (
                    f"Plugin '{self.name}' lacks '{cap.value}' capability. "
                    f"Add '{cap.value}' to capabilities in plugin.yaml."
                )
            )

    def register_tool(self, tool_instance: BaseTool) -> None:
        """Register a LangChain tool (requires the ``tools`` capability)."""
        self._check_capability(PluginCapability.TOOLS)

        from contextunity.router.modules.tools import register_tool

        register_tool(tool_instance)
        tool_name = getattr(tool_instance, "name", str(tool_instance))
        self._registered_tools.append(tool_name)
        logger.info("[Plugin:%s] Registered tool: %s", self.name, tool_name)

    def register_graph(
        self,
        name: str,
        builder: RunnableGraphFactory | ZeroArgGraphThunk,
    ) -> None:
        """Register a graph builder under *name* (requires the ``graphs`` capability)."""
        self._check_capability(PluginCapability.GRAPHS)

        from contextunity.router.core.registry import as_runnable_graph_factory, graph_registry

        graph_registry.register(name, as_runnable_graph_factory(builder))
        self._registered_graphs.append(name)
        logger.info("[Plugin:%s] Registered graph: %s", self.name, name)

    def register_connector(self, name: str, cls: type[BaseConnector]) -> None:
        """Register a connector class under *name* (requires the ``connectors`` capability)."""
        self._check_capability(PluginCapability.CONNECTORS)

        from contextunity.router.core.registry import register_connector

        # Call the decorator function directly
        _ = register_connector(name)(cls)
        self._registered_connectors.append(name)
        logger.info("[Plugin:%s] Registered connector: %s", self.name, name)

    def register_provider(self, name: str, cls: type[BaseProvider]) -> None:
        """Register a provider class under *name* (requires the ``providers`` capability)."""
        self._check_capability(PluginCapability.PROVIDERS)

        from contextunity.router.core.registry import register_provider

        _ = register_provider(name)(cls)
        self._registered_providers.append(name)
        logger.info("[Plugin:%s] Registered provider: %s", self.name, name)

    def register_transformer(self, name: str, cls: type[BaseTransformer]) -> None:
        """Register a transformer class under *name* (requires the ``transformers`` capability)."""
        self._check_capability(PluginCapability.TRANSFORMERS)

        from contextunity.router.core.registry import register_transformer

        _ = register_transformer(name)(cls)
        self._registered_transformers.append(name)
        logger.info("[Plugin:%s] Registered transformer: %s", self.name, name)

    # ---- Introspection -----------------------------------------------------

    def summary(self) -> JsonDict:
        """Return a summary of what this plugin registered."""
        summary: JsonDict = {
            "name": self.name,
            "version": self.version,
            "capabilities": [c.value for c in self._capabilities],
            "tools": list(self._registered_tools),
            "graphs": list(self._registered_graphs),
            "connectors": list(self._registered_connectors),
            "providers": list(self._registered_providers),
            "transformers": list(self._registered_transformers),
        }
        return summary


# ============================================================================
# Plugin loading
# ============================================================================

# Global registry of loaded plugins
_loaded_plugins: dict[str, PluginContext] = {}


def load_manifest(plugin_dir: Path) -> PluginManifest | None:
    """Parse and validate ``plugin.yaml`` (or ``.yml``) from *plugin_dir*; return ``None`` if absent."""
    manifest_path = plugin_dir / "plugin.yaml"
    if not manifest_path.exists():
        manifest_path = plugin_dir / "plugin.yml"
    if not manifest_path.exists():
        return None

    from contextunity.core.parsing import yaml_load

    with open(manifest_path) as f:
        raw = yaml_load(f)

    if not is_json_dict(raw):
        raise RouterPluginError(f"Invalid plugin manifest in {manifest_path}: expected mapping")

    return PluginManifest.model_validate(raw)


def _check_version_compatibility(manifest: PluginManifest) -> None:
    """Raise ``ValueError`` if the installed ``contextunity.router`` version violates the plugin’s semver constraints."""
    required = manifest.requires.get("contextunity.router")
    if not required:
        return

    from importlib.metadata import version as get_version

    from packaging.version import Version

    try:
        current = Version(get_version("contextunity.router"))
    except Exception:
        # If we can't determine version, skip check
        logger.debug("Could not determine contextunity.router version for plugin %s", manifest.name)
        return

    # Parse requirement like ">=0.9.0"
    if required.startswith(">="):
        min_version = Version(required[2:])
        if current < min_version:
            raise RouterPluginError(
                (
                    f"Plugin '{manifest.name}' requires contextunity.router>={min_version}, "
                    f"but {current} is installed"
                )
            )
    elif required.startswith("=="):
        exact = Version(required[2:])
        if current != exact:
            raise RouterPluginError(
                f"Plugin '{manifest.name}' requires contextunity.router=={exact}, but {current} is installed"
            )


def load_plugin(plugin_dir: Path) -> PluginContext | None:
    """Load a plugin from a directory with manifest."""
    manifest = load_manifest(plugin_dir)
    if manifest is None:
        return None

    if not manifest.enabled:
        logger.info("Plugin '%s' is disabled, skipping", manifest.name)
        return None

    # Check for duplicate
    if manifest.name in _loaded_plugins:
        logger.warning("Plugin '%s' already loaded, skipping duplicate", manifest.name)
        return _loaded_plugins[manifest.name]

    # Version check
    _check_version_compatibility(manifest)

    # Create context
    ctx = PluginContext(manifest, plugin_dir=plugin_dir)

    # Import entry point
    entry_path = plugin_dir / manifest.entry_point
    if not entry_path.exists():
        raise FileNotFoundError(f"Plugin '{manifest.name}' entry point not found: {entry_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_plugin_{manifest.name}", entry_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load plugin '{manifest.name}' from {entry_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Call on_load hook
    on_load = getattr(module, "on_load", None)
    if callable(on_load):
        _ = on_load(ctx)
    else:
        logger.debug("Plugin '%s' has no on_load() function, loaded module only", manifest.name)

    # Register
    _loaded_plugins[manifest.name] = ctx
    logger.info(
        "Loaded plugin '%s' v%s (capabilities: %s)",
        manifest.name,
        manifest.version,
        [c.value for c in ctx.capabilities],
    )

    return ctx


def get_loaded_plugins() -> dict[str, PluginContext]:
    """Return all loaded plugins."""
    return _loaded_plugins.copy()


def reset_plugins() -> None:
    """Reset the plugin registry (for testing)."""
    _loaded_plugins.clear()


__all__ = [
    "PluginCapability",
    "PluginContext",
    "PluginManifest",
    "get_loaded_plugins",
    "load_manifest",
    "load_plugin",
    "reset_plugins",
]
