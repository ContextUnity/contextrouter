"""Main configuration class that combines all config modules."""

from contextunity.core import get_contextunit_logger
from contextunity.core.config import ServiceConfig, SharedSecurityConfig, set_env_default
from pydantic import Field

from .models import LLMConfig, ModelsConfig, PrivacyConfig, RouterSection
from .providers import (
    AnthropicConfig,
    GoogleCSEConfig,
    GroqConfig,
    HuggingFaceHubConfig,
    InceptionConfig,
    LangfuseConfig,
    LocalOpenAIConfig,
    OpenAIConfig,
    OpenRouterConfig,
    PerplexityConfig,
    PluginsConfig,
    PostgresConfig,
    RunPodConfig,
    SerperConfig,
    VertexConfig,
)
from .security import SecurityConfig


class RouterConfig(ServiceConfig):
    """Main configuration class for contextunity.router."""

    # Server settings
    port: int = 50050
    debug: bool = False
    debug_graph_messages: bool = False
    debug_tools_messages: bool = False

    # Sub-configurations
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    router: RouterSection = Field(default_factory=RouterSection)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    security: SharedSecurityConfig = Field(default_factory=SecurityConfig)

    # Provider configurations
    vertex: VertexConfig = Field(default_factory=VertexConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    inception: InceptionConfig = Field(default_factory=InceptionConfig)
    runpod: RunPodConfig = Field(default_factory=RunPodConfig)
    hf_hub: HuggingFaceHubConfig = Field(default_factory=HuggingFaceHubConfig)
    local: LocalOpenAIConfig = Field(default_factory=LocalOpenAIConfig)
    google_cse: GoogleCSEConfig = Field(default_factory=GoogleCSEConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)

    perplexity: PerplexityConfig = Field(default_factory=PerplexityConfig)
    serper: SerperConfig = Field(default_factory=SerperConfig)


# ---- Runtime endpoint resolution ----


def _resolve_service_endpoints(config: RouterConfig) -> RouterConfig:
    """Return a config copy with runtime service endpoints resolved.

    Uses SharedConfig.*_url fields as configured_host source.
    Resolved endpoints are written back to the same *_url fields.
    """
    from contextunity.core.discovery import resolve_service_endpoint

    logger = get_contextunit_logger(__name__)

    brain_endpoint = resolve_service_endpoint(
        "brain", configured_host=config.brain_url, default_host="localhost:50051"
    )
    worker_endpoint = resolve_service_endpoint(
        "worker", configured_host=config.worker_url, default_host="localhost:50052"
    )
    shield_endpoint = resolve_service_endpoint(
        "shield", configured_host=config.shield_url, default_host="localhost:50054"
    )

    data = config.model_dump()

    data["brain_url"] = brain_endpoint
    data["worker_url"] = worker_endpoint
    data["shield_url"] = shield_endpoint

    logger.info(
        "Service endpoints resolved: brain=%s, worker=%s, shield=%s",
        brain_endpoint or "(none)",
        worker_endpoint or "(none)",
        shield_endpoint or "(disabled)",
    )
    return RouterConfig.model_validate(data)


# ---- Global config management ----


def load_config(config_path: str | None = None) -> RouterConfig:
    """Load router config through the unified config loader."""
    from contextunity.core.config import load_service_config

    # Force Vertex AI mode for langchain-google-genai / google-genai SDK.
    set_env_default("GOOGLE_GENAI_USE_VERTEXAI", "true")

    env_mappings = {
        # Debug flags
        "DEBUG_GRAPH_MESSAGES": "debug_graph_messages",
        "DEBUG_TOOLS_MESSAGES": "debug_tools_messages",
        "CU_ROUTER_DEBUG": "debug",
        "CU_ROUTER_PII_ENCRYPTION_TTL_SEC": "privacy.pii_encryption_ttl_seconds",
        # Model configuration
        "CU_ROUTER_DEFAULT_LLM": "models.default_llm",
        "CU_ROUTER_ALLOW_GLOBAL_FALLBACK": "models.allow_global_fallback",
        # Vertex configuration
        "VERTEX_PROJECT_ID": "vertex.project_id",
        "VERTEX_LOCATION": "vertex.location",
        "VERTEX_DISCOVERY_ENGINE_LOCATION": "vertex.discovery_engine_location",
        "VERTEX_DATA_STORE_LOCATION": "vertex.data_store_location",
        "VERTEX_CREDENTIALS_PATH": "vertex.credentials_path",
        # Brain configuration
        "CU_BRAIN_INDEX_TOOLS": "router.brain_index_tools",
        # Router server
        "ROUTER_INSTANCE_NAME": "router.instance_name",
        "GCS_DEFAULT_BUCKET": "router.gcs_default_bucket",
        # Providers
        "OPENAI_API_KEY": "openai.api_key",
        "OPENAI_ORGANIZATION": "openai.organization",
        "OPENAI_REASONING_EFFORT": "openai.reasoning_effort",
        "ANTHROPIC_API_KEY": "anthropic.api_key",
        "PERPLEXITY_API_KEY": "perplexity.api_key",
        "SERPER_API_KEY": "serper.api_key",
        "OPENROUTER_API_KEY": "openrouter.api_key",
        "OPENROUTER_BASE_URL": "openrouter.base_url",
        "GROQ_API_KEY": "groq.api_key",
        "GROQ_BASE_URL": "groq.base_url",
        "RUNPOD_API_KEY": "runpod.api_key",
        "RUNPOD_BASE_URL": "runpod.base_url",
        "HF_TOKEN": "hf_hub.api_key",
        "HF_BASE_URL": "hf_hub.base_url",
        "INCEPTION_API_KEY": "inception.api_key",
        "INCEPTION_BASE_URL": "inception.base_url",
        "INCEPTION_REASONING_EFFORT": "inception.reasoning_effort",
        "LOCAL_OLLAMA_BASE_URL": "local.ollama_base_url",
        "LOCAL_VLLM_BASE_URL": "local.vllm_base_url",
        "GOOGLE_CSE_ENABLED": "google_cse.enabled",
        "GOOGLE_CSE_API_KEY": "google_cse.api_key",
        "GOOGLE_CSE_CX": "google_cse.search_engine_id",
        # Langfuse
        "LANGFUSE_SECRET_KEY": "langfuse.secret_key",
        "LANGFUSE_PUBLIC_KEY": "langfuse.public_key",
        "LANGFUSE_HOST": "langfuse.host",
        "LANGFUSE_PROJECT_ID": "langfuse.project_id",
        "LANGFUSE_ENVIRONMENT": "langfuse.environment",
        "LANGFUSE_SERVICE_NAME": "langfuse.service_name",
    }

    config = load_service_config(
        RouterConfig,
        "router",
        env_mappings=env_mappings,
        config_path=config_path,
    )
    return _resolve_service_endpoints(config)


from contextunity.core.config import ServiceConfigRegistry  # noqa: E402

_registry = ServiceConfigRegistry(load_config)

get_core_config = _registry.get
set_core_config = _registry.set
reset_core_config = _registry.reset
