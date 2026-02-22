"""Router → Brain service token factory.

Provides a single-source ContextToken for all Router service calls to Brain.
Every BrainClient created by the Router should use this token.

Usage::

    from contextrouter.core.brain_token import get_brain_service_token

    token = get_brain_service_token()
    client = BrainClient(host=brain_host, mode="grpc", token=token)
"""

from __future__ import annotations

from contextcore.permissions import Permissions
from contextcore.tokens import mint_service_token

__all__ = ["get_brain_service_token"]

_PERMISSIONS = (
    Permissions.BRAIN_READ,
    Permissions.BRAIN_WRITE,
    Permissions.MEMORY_READ,
    Permissions.MEMORY_WRITE,
    Permissions.TRACE_WRITE,
)


def get_brain_service_token():
    """Return a cached ContextToken for Router → Brain calls.

    Grants the minimal set of permissions needed by the Router:
    - brain:read / brain:write — RAG, knowledge, traces
    - memory:read / memory:write — episodic + entity memory
    - trace:write — agent trace logging

    Token has a 1-hour TTL (managed by ``mint_service_token``).
    """
    return mint_service_token("router-brain-service", permissions=_PERMISSIONS)
