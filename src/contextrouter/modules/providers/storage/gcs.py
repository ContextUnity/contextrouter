"""GCS provider (Google Cloud Storage).

Upload/download files to GCS buckets.
Uses ContextUnit protocol for data transport.
"""

from __future__ import annotations

import logging
from typing import Any

from contextcore import ContextToken, ContextUnit
from contextcore.exceptions import ProviderError

from contextrouter.core.interfaces import BaseProvider, IRead, IWrite
from contextrouter.core.tokens import AccessManager

logger = logging.getLogger(__name__)


class GCSProvider(BaseProvider, IRead, IWrite):
    """Google Cloud Storage provider.

    Reads and writes blobs using ContextUnit payload:
        - payload.bucket: GCS bucket name
        - payload.path: blob path within the bucket
        - payload.content: file content (for write)
        - payload.content_type: MIME type (default: application/octet-stream)
    """

    def __init__(self, *, default_bucket: str = "") -> None:
        self._default_bucket = default_bucket
        self._access = AccessManager.from_core_config()
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage  # type: ignore[import-not-found]

                self._client = storage.Client()
            except ImportError:
                raise ProviderError(
                    "google-cloud-storage is not installed",
                    code="GCS_MISSING_DEPENDENCY",
                )
        return self._client

    async def read(
        self,
        query: str,
        *,
        limit: int = 1,
        filters: dict[str, Any] | None = None,
        token: ContextToken,
    ) -> list[ContextUnit]:
        """Read a blob from GCS.

        Args:
            query: Blob path (e.g., "exports/products.csv").
            filters: Optional dict with "bucket" key.
            token: Access token for authorization.
        """
        self._access.verify_read(token)

        bucket_name = (filters or {}).get("bucket", self._default_bucket)
        if not bucket_name:
            raise ProviderError("GCS bucket name is required", code="GCS_NO_BUCKET")

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(query)

            if not blob.exists():
                return []

            content = blob.download_as_bytes()
            unit = ContextUnit(
                payload={
                    "content": content.decode("utf-8", errors="replace"),
                    "bucket": bucket_name,
                    "path": query,
                    "content_type": blob.content_type or "application/octet-stream",
                    "size": blob.size,
                },
                provenance=["provider:gcs:read"],
            )
            return [unit]

        except Exception as e:
            logger.error("GCS read failed: bucket=%s path=%s error=%s", bucket_name, query, e)
            raise ProviderError(
                f"GCS read failed: {e}",
                code="GCS_READ_ERROR",
            ) from e

    async def write(self, data: ContextUnit, *, token: ContextToken) -> None:
        """Write a blob to GCS.

        Expects data.payload to contain:
            - content: str or bytes to upload
            - bucket: GCS bucket name (optional, uses default)
            - path: blob path
            - content_type: MIME type (optional)
        """
        self._access.verify_write(token)

        payload = data.payload or {}
        content = payload.get("content", "")
        bucket_name = payload.get("bucket", self._default_bucket)
        path = payload.get("path", "")
        content_type = payload.get("content_type", "application/octet-stream")

        if not bucket_name:
            raise ProviderError("GCS bucket name is required", code="GCS_NO_BUCKET")
        if not path:
            raise ProviderError("GCS blob path is required", code="GCS_NO_PATH")

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(path)

            if isinstance(content, bytes):
                blob.upload_from_string(content, content_type=content_type)
            else:
                blob.upload_from_string(
                    str(content).encode("utf-8"),
                    content_type=content_type,
                )

            logger.info(
                "GCS write: bucket=%s path=%s size=%d", bucket_name, path, len(str(content))
            )

        except ProviderError:
            raise
        except Exception as e:
            logger.error("GCS write failed: bucket=%s path=%s error=%s", bucket_name, path, e)
            raise ProviderError(
                f"GCS write failed: {e}",
                code="GCS_WRITE_ERROR",
            ) from e

    async def sink(self, unit: ContextUnit, *, token: ContextToken) -> Any:
        """Sink delegates to write."""
        await self.write(unit, token=token)
        return None


__all__ = ["GCSProvider"]
