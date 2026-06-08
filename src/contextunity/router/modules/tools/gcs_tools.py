"""GCS storage tools for Dispatcher Agent.

Exposes Google Cloud Storage operations as LangChain tools.
Uses the GCSProvider (ContextUnit protocol) under the hood.

Tools:
    gcs_upload   — Upload content to a GCS bucket
    gcs_download — Download content from a GCS bucket
    gcs_list     — List blobs in a GCS bucket
"""

from __future__ import annotations

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.sdk.payload import get_int, get_str

from contextunity.router.langchain_boundaries import tool
from contextunity.router.modules.tools import register_tool
from contextunity.router.modules.tools.schemas import GCSResult

logger = get_contextunit_logger(__name__)


def _get_default_bucket() -> str:
    """Get default GCS bucket from config."""
    from contextunity.router.core import get_core_config

    return get_core_config().router.gcs_default_bucket


@tool
async def gcs_upload(
    content: str,
    path: str,
    bucket: str = "",
    content_type: str = "text/plain",
) -> GCSResult:
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

    from contextunity.router.modules.providers.storage.gcs import GCSProvider

    _default = _get_default_bucket()
    if bucket and bucket != _default:
        return GCSResult(
            success=False,
            status="error",
            error=f"Access denied: cannot upload to arbitrary bucket '{bucket}'.",
        )
    bucket = _default

    if not bucket:
        return GCSResult(
            success=False,
            status="error",
            error="No GCS bucket specified. Set GCS_DEFAULT_BUCKET env.",
        )

    provider = GCSProvider(default_bucket=bucket)

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
        await provider.write(unit)
        gcs_uri = f"gs://{bucket}/{path}"
        return GCSResult(
            success=True,
            status="uploaded",
            gcs_uri=gcs_uri,
            path=path,
            bucket=bucket,
            size=len(content),
        )
    except Exception as e:
        logger.error("GCS upload failed: %s", e)
        return GCSResult(success=False, status="error", error=str(e))


@tool
async def gcs_download(
    path: str,
    bucket: str = "",
) -> GCSResult:
    """Download content from Google Cloud Storage.

    Use this tool to read files from GCS buckets.

    Args:
        path: Blob path in the bucket (e.g., "exports/report.csv").
        bucket: GCS bucket name. Uses default if not specified.

    Returns:
        Dictionary with content and metadata, or error.
    """

    from contextunity.router.modules.providers.storage.gcs import GCSProvider

    _default = _get_default_bucket()
    if bucket and bucket != _default:
        return GCSResult(
            success=False,
            status="error",
            error=f"Access denied: cannot download from arbitrary bucket '{bucket}'.",
        )
    bucket = _default

    if not bucket:
        return GCSResult(
            success=False,
            status="error",
            error="No GCS bucket specified. Set GCS_DEFAULT_BUCKET env.",
        )

    provider = GCSProvider(default_bucket=bucket)

    try:
        results = await provider.read(path, filters={"bucket": bucket})
        if not results:
            return GCSResult(success=True, status="not_found", path=path, bucket=bucket)

        unit = results[0]
        payload = unit.payload or {}
        return GCSResult(
            success=True,
            status="downloaded",
            content=get_str(payload, "content"),
            path=path,
            bucket=bucket,
            content_type=get_str(payload, "content_type"),
            size=get_int(payload, "size"),
        )
    except Exception as e:
        logger.error("GCS download failed: %s", e)
        return GCSResult(success=False, status="error", error=str(e))


@tool
async def gcs_list(
    prefix: str = "",
    bucket: str = "",
    max_results: int = 50,
) -> GCSResult:
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
        return GCSResult(
            success=False,
            status="error",
            error=f"Access denied: cannot list arbitrary bucket '{bucket}'.",
        )
    bucket = _default

    if not bucket:
        return GCSResult(
            success=False,
            status="error",
            error="No GCS bucket specified. Set GCS_DEFAULT_BUCKET env.",
        )

    try:
        from contextunity.router.modules.providers.storage.gcs import list_gcs_blobs

        items = list_gcs_blobs(bucket, prefix=prefix, max_results=max_results)

        return GCSResult(
            success=True,
            status="ok",
            bucket=bucket,
            prefix=prefix,
            count=len(items),
            blobs=items,
        )

    except ImportError:
        return GCSResult(
            success=False,
            status="error",
            error="google-cloud-storage is not installed",
        )
    except Exception as e:
        logger.error("GCS list failed: %s", e)
        return GCSResult(success=False, status="error", error=str(e))


# ── Auto-register tools ──────────────────────────────────────────

register_tool(gcs_upload, permission="storage:write")
register_tool(gcs_download, permission="storage:read")
register_tool(gcs_list, permission="storage:read")

logger.info("Registered 3 GCS storage tools")

__all__ = ["gcs_upload", "gcs_download", "gcs_list"]
