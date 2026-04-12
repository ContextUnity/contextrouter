"""gRPC client utilities for Router."""

from __future__ import annotations

from typing import Optional

from contextunity.core import ContextToken, WorkerClient, get_contextunit_logger

logger = get_contextunit_logger(__name__)

# Global client instances
_worker_client: Optional[WorkerClient] = None


async def get_worker_client(token: Optional[ContextToken] = None) -> WorkerClient:
    """Get or create Worker gRPC client.

    Args:
        token: Optional ContextToken for authorization

    Returns:
        WorkerClient instance
    """
    global _worker_client

    from contextunity.router.core import get_core_config

    worker_url = get_core_config().router.worker_grpc_endpoint

    if token:
        # Always return a fresh client for token-specific requests to avoid
        # cross-contamination and race conditions under concurrent execution.
        return WorkerClient(host=worker_url, mode="grpc", token=token)

    if _worker_client is None:
        _worker_client = WorkerClient(host=worker_url, mode="grpc", token=None)

    return _worker_client


__all__ = ["get_worker_client"]
