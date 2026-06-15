"""Resolve the active ContextToken for Router tool execution."""

from __future__ import annotations

from contextunity.core import ContextToken
from contextunity.core.exceptions import SecurityError


def resolve_tool_context_token() -> ContextToken:
    """Return the verified caller token for Brain-backed tools.

    Resolution order:
    1. gRPC ``VerifiedAuthContext`` from the service interceptor.
    2. Attenuated token from ``secure_node`` graph execution (contextvar).

    Raises:
        SecurityError: When no token is available (fail-closed).
    """
    from contextunity.core.authz.context import get_auth_context

    from contextunity.router.core.context import get_current_access_token

    auth_ctx = get_auth_context()
    if auth_ctx and auth_ctx.token:
        return auth_ctx.token

    token = get_current_access_token()
    if token is not None:
        return token

    raise SecurityError(
        "Brain tool requires an active ContextToken from gRPC auth or secure graph execution"
    )


__all__ = ["resolve_tool_context_token"]
