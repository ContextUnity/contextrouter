"""GCS provider (Google Cloud Storage).
Upload/download files to GCS buckets.
Uses ContextUnit protocol for data transport.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, override

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.exceptions import ProviderError
from contextunity.core.sdk.payload import get_str
from contextunity.core.types import JsonDict, is_object_iterable

from contextunity.router.core.interfaces import BaseProvider, IRead, IWrite
from contextunity.router.modules.tools.schemas import GCSBlobEntry

logger = get_contextunit_logger(__name__)


class GCSBlob(Protocol):
    content_type: str | None
    size: int | None

    def exists(self) -> bool: ...

    def download_as_bytes(self) -> bytes: ...

    def upload_from_string(
        self,
        data: str | bytes,
        content_type: str | None = None,
    ) -> None: ...


class GCSListBlob(Protocol):
    """Blob metadata returned by ``Bucket.list_blobs``."""

    name: str
    size: int | None
    content_type: str | None
    updated: object | None


class GCSBucket(Protocol):
    def blob(self, blob_name: str) -> GCSBlob: ...

    def list_blobs(
        self,
        *,
        prefix: str = "",
        max_results: int | None = None,
    ) -> Iterable[GCSListBlob]: ...


class GCSClient(Protocol):
    def bucket(self, bucket_name: str) -> GCSBucket: ...


def _safe_getattr(obj: object, name: str, default: object = "") -> object:
    return getattr(obj, name, default)


class _GCSBlobAdapter:
    _inner: object
    content_type: str | None
    size: int | None

    def __init__(self, inner: object) -> None:
        self._inner = inner
        ct: object = _safe_getattr(inner, "content_type", None)
        self.content_type = ct if isinstance(ct, str) else None
        sz: object = _safe_getattr(inner, "size", None)
        self.size = sz if isinstance(sz, int) else None

    def exists(self) -> bool:
        exists_fn: object = _safe_getattr(self._inner, "exists")
        if not callable(exists_fn):
            return False
        result: object = exists_fn()
        return bool(result)

    def download_as_bytes(self) -> bytes:
        download_fn: object = _safe_getattr(self._inner, "download_as_bytes")
        if not callable(download_fn):
            raise ProviderError(
                "GCS blob.download_as_bytes is not callable",
                code="GCS_READ_ERROR",
            )
        data: object = download_fn()
        return data if isinstance(data, bytes) else bytes(str(data), "utf-8")

    def upload_from_string(
        self,
        data: str | bytes,
        content_type: str | None = None,
    ) -> None:
        upload_fn: object = _safe_getattr(self._inner, "upload_from_string")
        if not callable(upload_fn):
            raise ProviderError(
                "GCS blob.upload_from_string is not callable",
                code="GCS_WRITE_ERROR",
            )
        _ = upload_fn(data, content_type=content_type)


class _GCSBucketAdapter:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def blob(self, blob_name: str) -> GCSBlob:
        blob_fn: object = _safe_getattr(self._inner, "blob")
        if not callable(blob_fn):
            raise ProviderError("GCS bucket.blob is not callable", code="GCS_MISSING_DEPENDENCY")
        blob_obj: object = blob_fn(blob_name)
        return _GCSBlobAdapter(blob_obj)

    def list_blobs(
        self,
        *,
        prefix: str = "",
        max_results: int | None = None,
    ) -> Iterable[GCSListBlob]:
        list_fn: object = _safe_getattr(self._inner, "list_blobs")
        if not callable(list_fn):
            raise ProviderError(
                "GCS bucket.list_blobs is not callable",
                code="GCS_MISSING_DEPENDENCY",
            )
        blobs_obj: object = list_fn(prefix=prefix, max_results=max_results)
        return _GCSListBlobIterable(blobs_obj)


class _GCSListBlobAdapter:
    name: str
    size: int | None
    content_type: str | None
    updated: object | None

    def __init__(self, inner: object) -> None:
        name_val: object = _safe_getattr(inner, "name", "")
        self.name = name_val if isinstance(name_val, str) else str(name_val)
        size_val: object = _safe_getattr(inner, "size", None)
        self.size = size_val if isinstance(size_val, int) else None
        ct_val: object = _safe_getattr(inner, "content_type", None)
        self.content_type = ct_val if isinstance(ct_val, str) else None
        self.updated = _safe_getattr(inner, "updated", None)


class _GCSListBlobIterable:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __iter__(self) -> Iterator[GCSListBlob]:
        if not is_object_iterable(self._inner):
            return iter(())
        for blob_obj in self._inner:
            yield _GCSListBlobAdapter(blob_obj)


class _GCSClientAdapter:
    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def bucket(self, bucket_name: str) -> GCSBucket:
        bucket_fn: object = _safe_getattr(self._inner, "bucket")
        if not callable(bucket_fn):
            raise ProviderError(
                "GCS client.bucket is not callable",
                code="GCS_MISSING_DEPENDENCY",
            )
        bucket_obj: object = bucket_fn(bucket_name)
        return _GCSBucketAdapter(bucket_obj)


def _load_gcs_client() -> GCSClient:
    import importlib

    storage = importlib.import_module("google.cloud.storage")
    client_factory_obj: object = getattr(storage, "Client", None)
    if not callable(client_factory_obj):
        raise ProviderError(
            "google.cloud.storage.Client is not callable",
            code="GCS_MISSING_DEPENDENCY",
        )
    inner_client: object = client_factory_obj()
    return _GCSClientAdapter(inner_client)


def _blob_entry(blob: GCSListBlob) -> GCSBlobEntry:
    """Convert a GCS blob object into a JSON-serialisable entry."""
    updated_val = blob.updated
    return {
        "name": blob.name,
        "size": blob.size,
        "content_type": blob.content_type,
        "updated": str(updated_val) if updated_val is not None else None,
    }


def list_gcs_blobs(
    bucket_name: str,
    *,
    prefix: str = "",
    max_results: int = 50,
) -> list[GCSBlobEntry]:
    """List blobs in a bucket using a lazily imported GCS client."""
    client = _load_gcs_client()
    bucket_obj = client.bucket(bucket_name)
    blob_iter = bucket_obj.list_blobs(prefix=prefix, max_results=max_results)

    items: list[GCSBlobEntry] = []
    for blob_obj in blob_iter:
        blob: GCSListBlob = blob_obj
        items.append(_blob_entry(blob))
    return items


class GCSProvider(BaseProvider, IRead, IWrite):
    """Google Cloud Storage provider.

    Reads and writes blobs using ContextUnit payload:
        - payload.bucket: GCS bucket name
        - payload.path: blob path within the bucket
        - payload.content: file content (for write)
        - payload.content_type: MIME type (default: application/octet-stream)
    """

    _default_bucket: str
    _client: GCSClient | None

    def __init__(self, *, default_bucket: str = "") -> None:
        """Store the default bucket name; GCS client is created lazily on first use."""
        self._default_bucket = default_bucket
        self._client = None

    def _get_client(self) -> GCSClient:
        """Lazily import and return the ``google.cloud.storage`` client.

        Raises:
            ProviderError: If ``google-cloud-storage`` is not installed.
        """
        if self._client is None:
            try:
                self._client = _load_gcs_client()
            except ImportError:
                raise ProviderError(
                    "google-cloud-storage is not installed",
                    code="GCS_MISSING_DEPENDENCY",
                )
        return self._client

    @override
    async def read(
        self,
        query: str,
        *,
        limit: int = 1,
        filters: JsonDict | None = None,
    ) -> list[ContextUnit]:
        """Read a blob from gcs."""
        _ = limit
        bucket_name = self._default_bucket
        if filters:
            raw_bucket = filters.get("bucket")
            if isinstance(raw_bucket, str) and raw_bucket.strip():
                bucket_name = raw_bucket
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

    @override
    async def write(self, data: ContextUnit) -> None:
        """Write a blob to GCS.

        Args:
            data: ContextUnit with payload containing content, bucket, path.
        """
        payload = data.payload or {}
        content = payload.get("content", "")
        bucket_name = get_str(payload, "bucket", self._default_bucket)
        path = get_str(payload, "path")
        content_type = get_str(payload, "content_type", "application/octet-stream")

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

    @override
    async def sink(self, unit: ContextUnit) -> None:
        """Sink delegates to write."""
        await self.write(unit)
        return None


__all__ = ["GCSProvider", "list_gcs_blobs"]
