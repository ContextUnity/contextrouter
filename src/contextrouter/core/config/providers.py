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
    discovery_engine_location: str = "global"
    data_store_location: str = ""  # optional override
    # Credentials can be loaded from:
    # 1. GOOGLE_APPLICATION_CREDENTIALS env var (ADC)
    # 2. `credentials_path` (explicit service account JSON)
    credentials_path: str = ""


class OpenAIConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    organization: str | None = None


class GoogleCSEConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    search_engine_id: str = ""


class LiteLLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    api_base: str | None = None


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
