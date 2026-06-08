"""Modular configuration system for contextunity.router."""

from contextunity.core.config import (
    get_bool_env,
    get_env,
)

# RAGConfig moved - check if still needed for retrieval
# from .ingestion import RAGConfig
from .main import (
    RouterConfig,
    get_core_config,
    load_config,
    reset_core_config,
    set_core_config,
)
from .models import (
    LLMConfig,
    ModelsConfig,
    RouterSection,
)

# ConfigPaths is already imported from .main above
from .providers import (
    AnthropicConfig,
    GoogleCSEConfig,
    InceptionConfig,
    LangfuseConfig,
    LocalOpenAIConfig,
    OpenAIConfig,
    OpenRouterConfig,
    PluginsConfig,
    PostgresConfig,
    VertexConfig,
)
from .security import (
    SecurityConfig,
    SecurityPoliciesConfig,
)

# Re-export for backward compatibility
__all__ = [
    # Main classes
    "RouterConfig",
    # Main functions
    "get_core_config",
    "load_config",
    "reset_core_config",
    "set_core_config",
    # Base utilities
    "get_env",
    "get_bool_env",
    # Model configs
    "ModelsConfig",
    "LLMConfig",
    "RouterSection",
    # Provider configs
    "VertexConfig",
    "OpenAIConfig",
    "AnthropicConfig",
    "OpenRouterConfig",
    "InceptionConfig",
    "LocalOpenAIConfig",
    "GoogleCSEConfig",
    "LangfuseConfig",
    "PluginsConfig",
    "PostgresConfig",
    # Ingestion configs removed - moved to contextunity.brain
    # "RAGConfig",
    # Security configs
    "SecurityConfig",
    "SecurityPoliciesConfig",
]
