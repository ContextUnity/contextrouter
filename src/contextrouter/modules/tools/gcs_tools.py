"""GCS storage tools for Dispatcher Agent.

Exposes Google Cloud Storage operations as LangChain tools.
Uses the GCSProvider (ContextUnit protocol) under the hood.

Tools:
    gcs_upload   — Upload content to a GCS bucket
    gcs_download — Download content from a GCS bucket
    gcs_list     — List blobs in a GCS bucket
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from contextrouter.modules.tools import register_tool

logger = logging.getLogger(__name__)


def _get_default_bucket() -> str:
    """Get default GCS bucket from config."""
    from contextrouter.core import get_core_config

    return get_core_config().router.gcs_default_bucket


@tool
async def gcs_upload(
    content: str,
    path: str,
    bucket: str = "",
    content_type: str = "text/plain",
) -> dict[str, Any]:
    """Upload content to Google Cloud Storage.

    Use this tool to save files, exports, reports, or any data to GCS.

    Args:
        content: Text content to upload.
        path: Blob path in the bucket (e.g., "exports/report.csv").
        bucket: GCS bucket name. Uses default if not specified.
        content_type: MIME type of the content.

    Returns:
        Dictionary with upload status and GCS URI.
    """
    from contextcore import ContextToken, ContextUnit

    from contextrouter.modules.providers.storage.gcs import GCSProvider

    _default = _get_default_bucket()
    if bucket and bucket != _default:
        return {
            "status": "error",
            "error": f"Access denied: cannot upload to arbitrary bucket '{bucket}'.",
        }
    bucket = _default

    if not bucket:
        return {"status": "error", "error": "No GCS bucket specified. Set GCS_DEFAULT_BUCKET env."}

    provider = GCSProvider(default_bucket=bucket)
    token = ContextToken(
        token_id="tool-gcs-upload",
        permissions=("storage:write",),
    )

    unit = ContextUnit(
        payload={
            "content": content,
            "bucket": bucket,
            "path": path,
            "content_type": content_type,
        },
        provenance=["tool:gcs_upload"],
    )

    try:
        await provider.write(unit, token=token)
        gcs_uri = f"gs://{bucket}/{path}"
        return {
            "status": "uploaded",
            "gcs_uri": gcs_uri,
            "path": path,
            "bucket": bucket,
            "size": len(content),
        }
    except Exception as e:
        logger.error("GCS upload failed: %s", e)
        return {"status": "error", "error": str(e)}


@tool
async def gcs_download(
    path: str,
    bucket: str = "",
) -> dict[str, Any]:
    """Download content from Google Cloud Storage.

    Use this tool to read files from GCS buckets.

    Args:
        path: Blob path in the bucket (e.g., "exports/report.csv").
        bucket: GCS bucket name. Uses default if not specified.

    Returns:
        Dictionary with content and metadata, or error.
    """
    from contextcore import ContextToken

    from contextrouter.modules.providers.storage.gcs import GCSProvider

    _default = _get_default_bucket()
    if bucket and bucket != _default:
        return {
            "status": "error",
            "error": f"Access denied: cannot download from arbitrary bucket '{bucket}'.",
        }
    bucket = _default

    if not bucket:
        return {"status": "error", "error": "No GCS bucket specified. Set GCS_DEFAULT_BUCKET env."}

    provider = GCSProvider(default_bucket=bucket)
    token = ContextToken(
        token_id="tool-gcs-download",
        permissions=("storage:read",),
    )

    try:
        results = await provider.read(path, token=token, filters={"bucket": bucket})
        if not results:
            return {"status": "not_found", "path": path, "bucket": bucket}

        unit = results[0]
        payload = unit.payload or {}
        return {
            "status": "downloaded",
            "content": payload.get("content", ""),
            "path": path,
            "bucket": bucket,
            "content_type": payload.get("content_type", ""),
            "size": payload.get("size", 0),
        }
    except Exception as e:
        logger.error("GCS download failed: %s", e)
        return {"status": "error", "error": str(e)}


@tool
async def gcs_list(
    prefix: str = "",
    bucket: str = "",
    max_results: int = 50,
) -> dict[str, Any]:
    """List blobs in a Google Cloud Storage bucket.

    Use this tool to browse available files in a GCS bucket.

    Args:
        prefix: Filter blobs by path prefix (e.g., "exports/").
        bucket: GCS bucket name. Uses default if not specified.
        max_results: Maximum number of blobs to return.

    Returns:
        Dictionary with list of blob names and metadata.
    """
    _default = _get_default_bucket()
    if bucket and bucket != _default:
        return {
            "status": "error",
            "error": f"Access denied: cannot list arbitrary bucket '{bucket}'.",
        }
    bucket = _default

    if not bucket:
        return {"status": "error", "error": "No GCS bucket specified. Set GCS_DEFAULT_BUCKET env."}

    try:
        from google.cloud import storage  # type: ignore[import-not-found]

        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blobs = bucket_obj.list_blobs(prefix=prefix, max_results=max_results)

        items = []
        for blob in blobs:
            items.append(
                {
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_type,
                    "updated": str(blob.updated) if blob.updated else None,
                }
            )

        return {
            "status": "ok",
            "bucket": bucket,
            "prefix": prefix,
            "count": len(items),
            "blobs": items,
        }

    except ImportError:
        return {"status": "error", "error": "google-cloud-storage is not installed"}
    except Exception as e:
        logger.error("GCS list failed: %s", e)
        return {"status": "error", "error": str(e)}


# ── Auto-register tools ──────────────────────────────────────────

_GCS_TOOLS = [gcs_upload, gcs_download, gcs_list]

for _t in _GCS_TOOLS:
    register_tool(_t)

logger.info("Registered %d GCS storage tools", len(_GCS_TOOLS))

__all__ = ["gcs_upload", "gcs_download", "gcs_list"]
