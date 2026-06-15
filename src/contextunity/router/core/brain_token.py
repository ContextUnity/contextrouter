"""Router → Brain service token factory.

Thin wrapper over ``contextunity.core.tokens.get_brain_service_token()``.
Every Router call must pass an explicit ``allowed_tenants`` scope — empty
tenant lists are rejected (fail-closed).

Usage::

    from contextunity.router.core.brain_token import get_brain_service_token

    token = get_brain_service_token(allowed_tenants=(tenant_id,))
    client = BrainClient(host=brain_host, mode="grpc", token=token)
"""

from __future__ import annotations

from collections.abc import Iterable

from contextunity.core import ContextToken
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import get_brain_service_token as _get_brain_service_token

__all__ = ["get_brain_service_token"]


def _normalize_tenant_scope(allowed_tenants: Iterable[str]) -> tuple[str, ...]:
    tenants: list[str] = []
    for tenant in allowed_tenants:
        stripped = tenant.strip()
        if stripped:
            tenants.append(stripped)
    return tuple(tenants)


def get_brain_service_token(*, allowed_tenants: Iterable[str]) -> ContextToken:
    """Return a cached ContextToken for Router → Brain calls.

    Args:
        allowed_tenants: Non-empty tenant scope for the Brain client/token cache key.

    Raises:
        SecurityError: When ``allowed_tenants`` is empty after normalization.
    """
    tenants = _normalize_tenant_scope(allowed_tenants)
    if not tenants:
        raise SecurityError(
            "Router→Brain service token requires explicit allowed_tenants; "
            "empty tenant scope is forbidden."
        )
    return _get_brain_service_token("router", allowed_tenants=tenants)
