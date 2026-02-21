"""Router → Brain service token factory.

Provides a single-source ContextToken for all Router service calls to Brain.
Every BrainClient created by the Router should use this token.

Usage::

    from contextrouter.core.brain_token import get_brain_service_token

    token = get_brain_service_token()
    client = BrainClient(host=brain_host, mode="grpc", token=token)
"""

from __future__ import annotations

from functools import lru_cache

from contextcore.permissions import Permissions
from contextcore.tokens import ContextToken

__all__ = ["get_brain_service_token"]


@lru_cache(maxsize=1)
def get_brain_service_token() -> ContextToken:
    """Return a cached ContextToken for Router → Brain calls.

    Grants the minimal set of permissions needed by the Router:
    - brain:read / brain:write — RAG, knowledge, traces
    - memory:read / memory:write — episodic + entity memory
    - trace:write — agent trace logging
    """
    return ContextToken(
        token_id="router-brain-service",
        permissions=(
            Permissions.BRAIN_READ,
            Permissions.BRAIN_WRITE,
            Permissions.MEMORY_READ,
            Permissions.MEMORY_WRITE,
            Permissions.TRACE_WRITE,
        ),
    )
