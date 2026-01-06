"""Modular configuration system for ContextRouter."""

from .base import (
    DEFAULT_READ_PERMISSION,
    DEFAULT_WRITE_PERMISSION,
    get_bool_env,
    get_env,
)
from .ingestion import (
    RAGConfig,
)
from .main import (
    Config,
    ConfigPaths,
    FlowConfig,
    get_core_config,
    set_core_config,
)
from .models import (
    LLMConfig,
    ModelsConfig,
    RouterConfig,
)
from .paths import (
    ConfigPaths as _ConfigPaths,  # Avoid duplicate
)
from .providers import (
    GoogleCSEConfig,
    LangfuseConfig,
    LiteLLMConfig,
    OpenAIConfig,
    PluginsConfig,
    VertexConfig,
)
from .security import (
    SecurityConfig,
    SecurityPoliciesConfig,
)

# Re-export for backward compatibility
__all__ = [
    # Main classes
    "Config",
    "FlowConfig",
    "ConfigPaths",
    # Main functions
    "get_core_config",
    "set_core_config",
    # Base utilities
    "get_env",
    "get_bool_env",
    "DEFAULT_READ_PERMISSION",
    "DEFAULT_WRITE_PERMISSION",
    # Model configs
    "ModelsConfig",
    "LLMConfig",
    "RouterConfig",
    # Provider configs
    "VertexConfig",
    "OpenAIConfig",
    "GoogleCSEConfig",
    "LiteLLMConfig",
    "LangfuseConfig",
    "PluginsConfig",
    # Ingestion configs
    "RAGConfig",
    # Security configs
    "SecurityConfig",
    "SecurityPoliciesConfig",
]
