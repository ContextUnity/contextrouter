"""Upload providers for ingestion pipeline.

Modular upload system supporting multiple cloud providers.
Currently implemented: GCloud (Vertex AI Search)
Future: Azure, AWS, etc.
"""

from __future__ import annotations

from ..settings import RagIngestionConfig
from .base import (
    UploadProvider,
    UploadResult,
)
from .gcloud import (
    GCloudUploadProvider,
)

__all__ = ["UploadProvider", "UploadResult", "GCloudUploadProvider", "get_provider"]

# Registry of available providers
_PROVIDERS: dict[str, type[UploadProvider]] = {
    "gcloud": GCloudUploadProvider,
}


def get_provider(config: RagIngestionConfig) -> UploadProvider:
    """Get upload provider instance based on config.

    Args:
        config: Full ingestion config (from load_config())

    Returns:
        Configured UploadProvider instance

    Raises:
        ValueError: If provider is unknown
    """
    provider_name = config.upload.provider

    if provider_name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Unknown upload provider '{provider_name}'. Available: {available}")

    provider_class = _PROVIDERS[provider_name]
    provider_config = config.upload.gcloud.model_dump()

    return provider_class(provider_config)
