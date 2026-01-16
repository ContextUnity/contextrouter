"""Core framework primitives for contextrouter.

This package is the long-term home for:
- configuration (Pydantic settings, layered sources)
- registries (agents/connectors/transformers/providers/models)
- shared interfaces and state models

During migration this module must remain non-breaking: existing production entry
points continue to live in `contextrouter.cortex.*` until the final cleanup phase.
"""

from __future__ import annotations

import importlib
import types

# Note: registry is imported dynamically via __getattr__ to avoid circular imports
from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.config import (
    Config,
    FlowConfig,
    get_bool_env,
    get_core_config,
    get_env,
    set_core_config,
)
from contextrouter.core.config.base import set_env_default
from contextrouter.core.flow_manager import FlowManager
from contextrouter.core.interfaces import (
    BaseAgent,
    BaseConnector,
    BaseProvider,
    BaseTransformer,
    IRead,
    IWrite,
)
from contextrouter.core.registry import agent_registry, graph_registry
from contextrouter.core.tokens import AccessManager, BiscuitToken, TokenBuilder
from contextrouter.core.types import UserCtx

__all__ = [
    # Kernel
    "BisquitEnvelope",
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
    "registry",  # Access via contextrouter.core.registry
    # Security
    "AccessManager",
    "BiscuitToken",
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
    importing them eagerly to keep `import contextrouter.core` lightweight.
    """
    if name in {"config", "exceptions", "env", "interfaces", "registry", "types"}:
        return importlib.import_module(f"contextrouter.core.{name}")
    raise AttributeError(name)
