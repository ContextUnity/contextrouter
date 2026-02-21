"""gRPC client utilities for Router."""

from __future__ import annotations

import logging
from typing import Optional

from contextcore import ContextToken, WorkerClient

logger = logging.getLogger(__name__)

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

    from contextrouter.core import get_core_config

    worker_url = get_core_config().router.worker_grpc_endpoint

    if _worker_client is None:
        _worker_client = WorkerClient(host=worker_url, mode="grpc", token=token)
    elif token and _worker_client.token != token:
        # Update token if different
        _worker_client = WorkerClient(host=worker_url, mode="grpc", token=token)

    return _worker_client


__all__ = ["get_worker_client"]
