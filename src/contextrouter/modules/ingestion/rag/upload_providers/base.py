"""Abstract base class for upload providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UploadResult:
    """Result of an upload operation."""

    success: bool
    provider: str
    # Provider-specific details
    details: dict[str, Any]
    error: str | None = None


class UploadProvider(ABC):
    """Abstract base class for upload providers.

    Subclasses implement upload logic for specific cloud platforms:
    - GCloud: GCS + Vertex AI Search
    - Azure: Blob Storage + Azure AI Search (future)
    - AWS: S3 + OpenSearch/Kendra (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'gcloud', 'azure', 'aws')."""
        ...

    @abstractmethod
    def upload_and_index(
        self,
        local_path: Path,
        *,
        wait: bool = False,
    ) -> UploadResult:
        """Upload file and trigger indexing.

        Args:
            local_path: Path to local JSONL file
            wait: If True, wait for indexing to complete

        Returns:
            UploadResult with operation details
        """
        ...

    @abstractmethod
    def get_config_summary(self) -> dict[str, str]:
        """Get summary of current configuration for logging."""
        ...
