"""Router → Brain service token factory.

Thin wrapper over contextcore.tokens.get_brain_service_token().
Kept for backward-compatible imports across the Router codebase.

Usage::

    from contextrouter.core.brain_token import get_brain_service_token

    token = get_brain_service_token()
    client = BrainClient(host=brain_host, mode="grpc", token=token)
"""

from __future__ import annotations

from contextcore.tokens import get_brain_service_token as _get_brain_service_token

__all__ = ["get_brain_service_token"]


def get_brain_service_token():
    """Return a cached ContextToken for Router → Brain calls."""
    return _get_brain_service_token("router")
