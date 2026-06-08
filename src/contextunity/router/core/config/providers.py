"""Provider configurations for external services."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class VertexConfig(BaseModel):
    """Google Cloud Vertex AI configuration (LLM and Discovery Engine)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    project_id: str = ""
    location: str = "us-central1"
    # Discovery Engine / Vertex AI Search is typically "global" even when Vertex AI LLM is regional.
    # Keep this separate to avoid accidental 0-result queries due to wrong location.
    #
    # Preferred names:
    # - discovery_engine_location: location for Discovery Engine (Vertex AI Search)
    # - data_store_location: alias for orgs that think in datastore terms
    discovery_engine_location: str = "global"  # Default, can be overridden by env
    data_store_location: str = ""  # optional override
    # Credentials can be loaded from:
    # 1. GOOGLE_APPLICATION_CREDENTIALS env var (ADC)
    # 2. `credentials_path` (explicit service account JSON)
    credentials_path: str = ""


class GeminiConfig(BaseModel):
    """Google AI Studio (Gemini) API configuration.

    Uses google-genai SDK with API key auth (not Vertex AI / service account).
    Get a key at https://aistudio.google.com/apikey
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    default_model: str = "gemini-2.5-flash"


class PostgresConfig(BaseModel):
    """PostgreSQL connection and vector storage configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    dsn: str = ""
    pool_min_size: int = 2
    pool_max_size: int = 10
    rls_enabled: bool = True
    vector_dim: int = 768


class OpenAIConfig(BaseModel):
    """OpenAI API configuration (GPT and reasoning models)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    organization: str | None = None
    # Reasoning effort for reasoning models (gpt-5, o1, o3)
    # Options: "minimal", "low", "medium", "high"
    # "minimal" saves tokens, "high" uses more reasoning
    reasoning_effort: str = "minimal"


class AnthropicConfig(BaseModel):
    """Anthropic API configuration (Claude models)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""


class PerplexityConfig(BaseModel):
    """Perplexity API configuration (LLM with search)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    default_model: str = "sonar"


class SerperConfig(BaseModel):
    """Serper API configuration (Google Search)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration (multi-provider gateway)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"


class LocalOpenAIConfig(BaseModel):
    """Base URLs for local OpenAI-compatible servers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    ollama_base_url: str = "http://localhost:11434/v1"
    vllm_base_url: str = "http://localhost:8000/v1"


class GroqConfig(BaseModel):
    """Groq API configuration (LPU-accelerated inference)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"


class InceptionConfig(BaseModel):
    """Inception Labs API configuration (Mercury-2 diffusion LLM).

    Docs: https://docs.inceptionlabs.ai/get-started/models
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = "https://api.inceptionlabs.ai/v1"
    # Reasoning effort: instant, low, medium, high (default: medium on Mercury-2 side)
    reasoning_effort: str = ""


class RunPodConfig(BaseModel):
    """RunPod serverless endpoint configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""
    # Usually: https://api.runpod.ai/v2/<endpoint_id>/openai/v1
    base_url: str = ""


class HuggingFaceHubConfig(BaseModel):
    """Hugging Face Inference API configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    api_key: str = ""  # HF_TOKEN
    base_url: str = "https://api-inference.huggingface.co/v1"


class GoogleCSEConfig(BaseModel):
    """Google Programmable Search Engine (CSE) configuration."""

    # Support `cx` key in TOML (`google_cse.cx`) while keeping a more descriptive field name internally.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = False
    api_key: str = ""
    search_engine_id: str = Field(default="", alias="cx")

    @property
    def cx(self) -> str:
        """Compatibility alias for ``search_engine_id``.

        Returns:
            The configured search engine ID.
        """
        return self.search_engine_id

    @cx.setter
    def cx(self, v: str) -> None:
        """Set ``search_engine_id`` via the ``cx`` alias.

        Args:
            v: New search engine ID value.
        """
        self.search_engine_id = v


class LangfuseConfig(BaseModel):
    """Langfuse observability platform configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    secret_key: str = ""
    public_key: str = ""
    host: str = "https://cloud.langfuse.com"
    project_id: str = ""  # Default project ID for dashboard URL construction
    environment: str = "development"
    service_name: str = "contextunity.router"


class PluginsConfig(BaseModel):
    """User plugin directories to scan eagerly (explicit opt-in)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    paths: list[str] = Field(default_factory=list)
