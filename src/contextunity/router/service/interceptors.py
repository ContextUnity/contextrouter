"""gRPC interceptor for contextunity.router permission enforcement.
Maps each Router RPC method to the exact permission required
and validates the ContextToken carries that permission.
Delegates to ``contextunity.core.security.ServicePermissionInterceptor``
for unified enforcement logic. Router only owns the RPC_PERMISSION_MAP.
Convention:
    ``""`` (empty string) = identity-verified only.  The interceptor still
    validates the token (crypto, expiry, revocation) but does **not** require
    a specific permission.  The RPC handler is responsible for its own
    authorization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextunity.core.permissions import Permissions
from contextunity.core.security import ServicePermissionInterceptor

if TYPE_CHECKING:
    from contextunity.router.core.config.main import RouterConfig

# ── RPC → Permission mapping ──────────────────────────────────

RPC_PERMISSION_MAP: dict[str, str] = {
    # Execution
    "ExecuteAgent": Permissions.ROUTER_EXECUTE,
    "StreamAgent": Permissions.ROUTER_EXECUTE,
    "ExecuteDispatcher": Permissions.ROUTER_EXECUTE,
    "StreamDispatcher": Permissions.ROUTER_EXECUTE,
    # Node Execution (Worker → Router callback)
    "ExecuteNode": Permissions.ROUTER_EXECUTE_NODE,
    # Registration & federated execution: handler-managed auth.
    # RegisterManifest requires tools:register:<project_id> inside the handler
    # after it reads the bundle project_id.
    "RegisterManifest": "",
    "ToolExecutorStream": "",
    # Introspection (read-only, sanitized)
    "IntrospectRegistrations": Permissions.ROUTER_INTROSPECT,
}


class RouterPermissionInterceptor(ServicePermissionInterceptor):
    """Router-specific permission interceptor.

    Thin wrapper around ``ServicePermissionInterceptor`` that pre-fills
    the Router RPC permission map and service name.

    Usage::

        interceptor = RouterPermissionInterceptor()
        server = grpc.aio.server(interceptors=[interceptor])
    """

    def __init__(self, *, shield_url: str = "", config: "RouterConfig | None" = None) -> None:
        """Pre-fill the Router RPC permission map and service name."""
        super().__init__(
            RPC_PERMISSION_MAP,
            service_name="Router",
            shield_url=shield_url,
            config=config,
        )


__all__ = [
    "RouterPermissionInterceptor",
    "RPC_PERMISSION_MAP",
]
