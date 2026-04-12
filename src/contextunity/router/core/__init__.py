"""Core framework primitives for cu.router.

This package is the long-term home for:
- configuration (Pydantic settings, layered sources)
- registries (agents/connectors/transformers/providers/models)
- shared interfaces and state models

During migration this module must remain non-breaking: existing production entry
points continue to live in `cu.router.cortex.*` until the final cleanup phase.
"""

from __future__ import annotations

import importlib
import types

# Note: registry is imported dynamically via __getattr__ to avoid circular imports
from contextunity.core import ContextUnit
from contextunity.core.tokens import ContextToken, TokenBuilder

from contextunity.router.core.config import (
    Config,
    FlowConfig,
    get_bool_env,
    get_core_config,
    get_env,
    set_core_config,
)
from contextunity.router.core.config.base import set_env_default
from contextunity.router.core.flow_manager import FlowManager
from contextunity.router.core.interfaces import (
    BaseAgent,
    BaseConnector,
    BaseProvider,
    BaseTransformer,
    IRead,
    IWrite,
)
from contextunity.router.core.registry import agent_registry, graph_registry
from contextunity.router.core.types import UserCtx

__all__ = [
    # Kernel
    "ContextUnit",
    "Config",
    "FlowConfig",
    "get_core_config",
    "set_core_config",
    "get_env",
    "get_bool_env",
    "set_env_default",
    "FlowManager",
    # Interfaces
    "BaseAgent",
    "BaseConnector",
    "BaseProvider",
    "BaseTransformer",
    "IRead",
    "IWrite",
    # Registry
    "agent_registry",  # Direct access for compatibility
    "graph_registry",  # Direct access for compatibility
    "registry",  # Access via cu.router.core.registry
    # Plugins
    "plugins",  # Plugin manifest and context system
    # Security
    "ContextToken",
    "TokenBuilder",
    # Types
    "UserCtx",
    # Modules (for backward compatibility)
    "config",
    "exceptions",
    "env",
    "interfaces",
    "registry",
    "types",
]


def __getattr__(name: str) -> types.ModuleType:
    """Lazy module attributes for backward compatibility.

    These names are listed in __all__ for historical reasons, but we avoid
    importing them eagerly to keep `import contextunity.router.core` lightweight.
    """
    if name in {"config", "exceptions", "env", "interfaces", "plugins", "registry", "types"}:
        return importlib.import_module(f"contextunity.router.core.{name}")
    raise AttributeError(name)
