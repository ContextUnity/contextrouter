"""Main configuration class that combines all config modules."""

import logging
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from .base import get_bool_env, get_env, set_env_default
from .ingestion import RAGConfig
from .models import LLMConfig, ModelsConfig, RouterConfig
from .paths import ConfigPaths
from .providers import (
    GoogleCSEConfig,
    LangfuseConfig,
    OpenAIConfig,
    PluginsConfig,
    VertexConfig,
)
from .security import SecurityConfig


class FlowConfig(BaseModel):
    """Configuration for a specific data processing flow.

    This is used by the flow manager to execute custom processing pipelines.
    """

    model_config = ConfigDict(extra="ignore")

    # Flow identification
    name: str = ""
    description: str = ""

    # Source configuration
    source: str = ""  # e.g., "web", "file", "api"
    source_params: dict[str, Any] = Field(default_factory=dict)

    # Processing pipeline
    logic: list[str] = Field(default_factory=list)  # Transformer names
    logic_params: dict[str, Any] = Field(default_factory=dict)

    # Sink configuration
    sink: str = ""  # e.g., "vertex", "postgres"
    sink_params: dict[str, Any] = Field(default_factory=dict)

    # Execution controls
    overwrite: bool = True
    workers: int = 1


class Config(BaseModel):
    """Main configuration class for ContextRouter.

    This combines all configuration modules into a single, hierarchical structure.
    Configuration is loaded from multiple sources in priority order:
    1. Environment variables
    2. TOML configuration files
    3. Default values
    """

    model_config = ConfigDict(extra="ignore")

    # Core settings
    debug: bool = False
    log_level: str = "INFO"

    # Sub-configurations
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    ingestion: RAGConfig = Field(default_factory=RAGConfig)

    # Provider configurations
    vertex: VertexConfig = Field(default_factory=VertexConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    google_cse: GoogleCSEConfig = Field(default_factory=GoogleCSEConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)

    # Internal state
    paths_cache: ConfigPaths | None = None
    loaded_from: list[Path] = Field(default_factory=list)

    @property
    def paths(self) -> ConfigPaths:
        """Get configuration paths."""
        if self.paths_cache is None:
            # Try to find project root
            from pathlib import Path

            # Look for pyproject.toml or go up from current file
            current = Path(__file__).resolve()
            for parent in [current.parent] + list(current.parents):
                if (parent / "pyproject.toml").exists():
                    self.paths_cache = ConfigPaths.from_root(parent)
                    break
            else:
                # Fallback to current directory
                self.paths_cache = ConfigPaths.from_root(Path.cwd())

        return self.paths_cache

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> "Config":
        """Load configuration from files and environment."""
        load_dotenv()

        # Optional explicit override for core config path.
        # This is intentionally separate from ingestion's CONTEXTROUTER_CONFIG_PATH.
        core_config_path = get_env("CONTEXTROUTER_CORE_CONFIG_PATH")

        # Force Vertex AI mode for langchain-google-genai / google-genai SDK.
        # Without this, the SDK may try API-key auth and fail with:
        # "Could not resolve API token from the environment".
        # Must be set before any ChatGoogleGenerativeAI instance is created.
        set_env_default("GOOGLE_GENAI_USE_VERTEXAI", "true")

        config = cls()
        paths = config.paths

        # Load from TOML if available
        toml_path = (
            Path(config_path)
            if config_path
            else (Path(core_config_path).resolve() if core_config_path else paths.toml_config)
        )
        if toml_path.exists():
            try:
                with open(toml_path, "rb") as f:
                    toml_data = tomllib.load(f)

                # Remove read-only properties (like 'paths')
                toml_data.pop("paths", None)

                # Merge TOML data with defaults using Pydantic
                config_dict = config.model_dump()
                config_dict.update(toml_data)
                config = cls.model_validate(config_dict)

                # Restore paths_cache to avoid recomputation
                config.paths_cache = paths
                config.loaded_from.append(toml_path)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load TOML config from {toml_path}: {e}")

        # Override with environment variables
        config._apply_env_overrides()

        return config

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        # Model configuration
        if llm_val := get_env("CONTEXTROUTER_DEFAULT_LLM"):
            self.models.default_llm = llm_val
        if intent_val := get_env("CONTEXTROUTER_INTENT_LLM"):
            self.models.intent_llm = intent_val
        if suggestions_val := get_env("CONTEXTROUTER_SUGGESTIONS_LLM"):
            self.models.suggestions_llm = suggestions_val
        if generation_val := get_env("CONTEXTROUTER_GENERATION_LLM"):
            self.models.generation_llm = generation_val
        if no_results_val := get_env("CONTEXTROUTER_NO_RESULTS_LLM"):
            self.models.no_results_llm = no_results_val

        # Vertex configuration
        if project_id := get_env("CONTEXTROUTER_VERTEX_PROJECT_ID"):
            self.vertex.project_id = project_id
        if location := get_env("CONTEXTROUTER_VERTEX_LOCATION"):
            self.vertex.location = location
        # Vertex AI Search / Discovery Engine location (separate from Vertex LLM region).
        if v := get_env("CONTEXTROUTER_VERTEX_DISCOVERY_ENGINE_LOCATION"):
            self.vertex.discovery_engine_location = v
        if v := get_env("CONTEXTROUTER_VERTEX_DATA_STORE_LOCATION"):
            self.vertex.data_store_location = v
        if credentials_path := get_env("CONTEXTROUTER_VERTEX_CREDENTIALS_PATH"):
            self.vertex.credentials_path = credentials_path

        # OpenAI configuration
        if openai_key := get_env("OPENAI_API_KEY"):
            self.openai.api_key = openai_key
        if openai_org := get_env("OPENAI_ORGANIZATION"):
            self.openai.organization = openai_org

        # Google CSE configuration
        if cse_enabled := get_bool_env("GOOGLE_CSE_ENABLED"):
            self.google_cse.enabled = cse_enabled
        if cse_key := get_env("GOOGLE_CSE_API_KEY"):
            self.google_cse.api_key = cse_key
        if cse_cx := get_env("GOOGLE_CSE_CX"):
            self.google_cse.search_engine_id = cse_cx

        # Langfuse configuration
        if langfuse_secret := get_env("LANGFUSE_SECRET_KEY"):
            self.langfuse.secret_key = langfuse_secret
        if langfuse_public := get_env("LANGFUSE_PUBLIC_KEY"):
            self.langfuse.public_key = langfuse_public
        if langfuse_host := get_env("LANGFUSE_HOST"):
            self.langfuse.host = langfuse_host
        if langfuse_env := get_env("LANGFUSE_ENVIRONMENT"):
            self.langfuse.environment = langfuse_env
        if langfuse_service := get_env("LANGFUSE_SERVICE_NAME"):
            self.langfuse.service_name = langfuse_service

        # Security configuration
        if security_enabled := get_bool_env("CONTEXTROUTER_SECURITY_ENABLED"):
            self.security.enabled = security_enabled
        if private_key_path := get_env("CONTEXTROUTER_PRIVATE_KEY_PATH"):
            self.security.private_key_path = private_key_path

        # Debug/Logging
        if debug_val := get_bool_env("CONTEXTROUTER_DEBUG"):
            self.debug = debug_val
        if log_level := get_env("CONTEXTROUTER_LOG_LEVEL"):
            self.log_level = log_level


# ---- Global config management ----

_GLOBAL_CONFIG: Config | None = None


def get_core_config() -> Config:
    """Return process-global core config (for framework modules)."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = Config.load()
    return _GLOBAL_CONFIG


def set_core_config(config: Config) -> None:
    """Set the global core config."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config
