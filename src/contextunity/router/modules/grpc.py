"""gRPC client utilities -- channel creation, stub caching, and health-check helpers for Router."""

from __future__ import annotations

from contextunity.core import ContextToken, WorkerClient, get_contextunit_logger

logger = get_contextunit_logger(__name__)

# Global client instances
_worker_client: WorkerClient | None = None


async def get_worker_client(token: ContextToken | None = None) -> WorkerClient:
    """Get or create Worker gRPC client.

    Args:
        token: Optional ContextToken for authorization

    Returns:
        WorkerClient instance
    """
    global _worker_client

    from contextunity.router.core import get_core_config

    worker_url = get_core_config().worker_url

    if token:
        # Always return a fresh client for token-specific requests to avoid
        # cross-contamination and race conditions under concurrent execution.
        return WorkerClient(host=worker_url, token=token)

    if _worker_client is None:
        _worker_client = WorkerClient(host=worker_url, token=None)

    return _worker_client


__all__ = ["get_worker_client"]
