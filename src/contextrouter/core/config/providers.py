"""Provider configurations for external services."""

from pydantic import BaseModel, ConfigDict, Field


class VertexConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


class PostgresConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dsn: str = ""
    pool_min_size: int = 2
    pool_max_size: int = 10
    rls_enabled: bool = True
    vector_dim: int = 768


class OpenAIConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    organization: str | None = None


class AnthropicConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""


class OpenRouterConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"


class LocalOpenAIConfig(BaseModel):
    """Base URLs for local OpenAI-compatible servers."""

    model_config = ConfigDict(extra="ignore")

    ollama_base_url: str = "http://localhost:11434/v1"
    vllm_base_url: str = "http://localhost:8000/v1"


class GroqConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"


class RunPodConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    # Usually: https://api.runpod.ai/v2/<endpoint_id>/openai/v1
    base_url: str = ""


class HuggingFaceHubConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""  # HF_TOKEN
    base_url: str = "https://api-inference.huggingface.co/v1"


class GoogleCSEConfig(BaseModel):
    # Support `cx` key in TOML (`google_cse.cx`) while keeping a more descriptive field name internally.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = False
    api_key: str = ""
    search_engine_id: str = Field(default="", alias="cx")

    @property
    def cx(self) -> str:
        """Compatibility alias for Google CSE search engine id (cx)."""
        return self.search_engine_id

    @cx.setter
    def cx(self, v: str) -> None:
        self.search_engine_id = v


class LangfuseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    secret_key: str = ""
    public_key: str = ""
    host: str = "https://cloud.langfuse.com"
    environment: str = "development"
    service_name: str = "contextrouter"


class PluginsConfig(BaseModel):
    """User plugin directories to scan eagerly (explicit opt-in)."""

    model_config = ConfigDict(extra="ignore")
    paths: list[str] = Field(default_factory=list)
