"""Google Cloud upload provider: GCS + Vertex AI Search."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextrouter.core import get_core_config, get_env
from contextrouter.modules.retrieval.rag.settings import resolve_data_store_id

from .base import (
    UploadProvider,
    UploadResult,
)

logger = logging.getLogger(__name__)


class GCloudUploadProvider(UploadProvider):
    """Upload provider for Google Cloud (GCS + Vertex AI Search).

    Configuration priority (highest to lowest):
    1. Explicit config values in settings.toml [upload.gcloud]
    2. Environment variables (VERTEX_PROJECT_ID, VERTEX_LOCATION, RAG_GCS_BUCKET, RAG_DB_NAME)

    Supports blue/green symbolic names for data_store_id.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize GCloud provider.

        Args:
            config: Provider-specific config from [upload.gcloud] section
        """
        # Ensure environment is loaded
        get_core_config()
        self._config = config

        # Resolve configuration with fallback to env vars
        self._project_id = (
            config.get("project_id")
            or get_env("VERTEX_PROJECT_ID")
            or get_env("CONTEXTROUTER_VERTEX_PROJECT_ID")
        )
        self._location = (
            config.get("location")
            or get_env("VERTEX_LOCATION", "global")
            or get_env("CONTEXTROUTER_VERTEX_LOCATION", "global")
        )
        self._gcs_bucket = config.get("gcs_bucket") or get_env("RAG_GCS_BUCKET")

        # Data store ID: config > resolve from env
        config_ds_id = config.get("data_store_id")
        if config_ds_id:
            # Config value might be symbolic ("blue"/"green") or actual ID
            self._data_store_id = self._resolve_symbolic(config_ds_id)
        else:
            # Fall back to env-based resolution
            self._data_store_id = None  # Will resolve at upload time

    def _resolve_symbolic(self, ds_id: str) -> str:
        """Resolve symbolic blue/green to actual datastore ID."""
        ds_lower = ds_id.lower().strip()
        if ds_lower in ("blue", "green"):
            return resolve_data_store_id(ds_lower)
        return ds_id

    @property
    def name(self) -> str:
        return "gcloud"

    def _get_data_store_id(self) -> str:
        """Get data store ID, resolving at runtime if needed."""
        if self._data_store_id:
            return self._data_store_id
        return resolve_data_store_id()

    def upload_and_index(
        self,
        local_path: Path,
        *,
        wait: bool = False,
    ) -> UploadResult:
        """Upload JSONL to GCS and trigger Vertex AI Search import."""
        # Lazy imports for optional GCP dependencies
        from google.api_core.client_options import ClientOptions
        from google.cloud import discoveryengine_v1 as discoveryengine
        from google.cloud import storage

        # Validate required config
        if not self._project_id:
            return UploadResult(
                success=False,
                provider=self.name,
                details={},
                error="project_id not set (config or VERTEX_PROJECT_ID env)",
            )
        if not self._gcs_bucket:
            return UploadResult(
                success=False,
                provider=self.name,
                details={},
                error="gcs_bucket not set (config or RAG_GCS_BUCKET env)",
            )
        if not local_path.exists():
            return UploadResult(
                success=False,
                provider=self.name,
                details={},
                error=f"File not found: {local_path}",
            )

        try:
            data_store_id = self._get_data_store_id()
        except ValueError as e:
            return UploadResult(
                success=False,
                provider=self.name,
                details={},
                error=f"Failed to resolve data_store_id: {e}",
            )

        logger.info(
            "GCloud upload: project=%s, bucket=%s, datastore=%s, location=%s",
            self._project_id,
            self._gcs_bucket,
            data_store_id,
            self._location,
        )

        try:
            # Folder by date, files keep original names (e.g., import_20251229_green_video.jsonl)
            date_folder = datetime.now(timezone.utc).strftime("%Y%m%d")
            blob_name = f"ingestion/{date_folder}/{local_path.name}"

            # Upload to GCS
            logger.info(
                "Uploading %s to gs://%s/%s ...", local_path.name, self._gcs_bucket, blob_name
            )
            storage_client = storage.Client()
            bucket = storage_client.bucket(self._gcs_bucket)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path), content_type="application/json")
            gcs_uri = f"gs://{self._gcs_bucket}/{blob_name}"
            logger.info("Uploaded to %s", gcs_uri)

            # Trigger Vertex AI Search import
            logger.info(
                "Triggering import to datastore '%s' in location '%s'...",
                data_store_id,
                self._location,
            )

            client_options = (
                ClientOptions(api_endpoint=f"{self._location}-discoveryengine.googleapis.com")
                if self._location != "global"
                else None
            )
            de_client = discoveryengine.DocumentServiceClient(client_options=client_options)

            parent = de_client.branch_path(
                project=self._project_id,
                location=self._location,
                data_store=data_store_id,
                branch="default_branch",
            )

            request = discoveryengine.ImportDocumentsRequest(
                parent=parent,
                gcs_source=discoveryengine.GcsSource(input_uris=[gcs_uri]),
                reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
            )

            operation = de_client.import_documents(request=request)
            op_name = operation.operation.name
            logger.info("Import operation started: %s", op_name)

            if wait:
                logger.info("Waiting for import to complete (this may take several minutes)...")
                response = operation.result(timeout=3600)  # 1 hour timeout
                logger.info("Import completed successfully!")
                logger.info("Import result: %s", response)

            return UploadResult(
                success=True,
                provider=self.name,
                details={
                    "operation_name": op_name,
                    "gcs_uri": gcs_uri,
                    "data_store_id": data_store_id,
                    "date": date_folder,
                    "project_id": self._project_id,
                    "location": self._location,
                },
            )

        except Exception as e:
            logger.exception("GCloud upload failed")
            return UploadResult(
                success=False,
                provider=self.name,
                details={},
                error=str(e),
            )

    def get_config_summary(self) -> dict[str, str]:
        """Get summary of current configuration for logging."""
        try:
            ds_id = self._get_data_store_id()
        except ValueError:
            ds_id = "<unresolved>"

        return {
            "provider": self.name,
            "project_id": self._project_id or "<not set>",
            "location": self._location,
            "gcs_bucket": self._gcs_bucket or "<not set>",
            "data_store_id": ds_id,
        }
