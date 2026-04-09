"""gRPC interceptor for ContextRouter permission enforcement.

Maps each Router RPC method to the exact permission required
and validates the ContextToken carries that permission.

Delegates to ``contextcore.security.ServicePermissionInterceptor``
for unified enforcement logic. Router only owns the RPC_PERMISSION_MAP.
"""

from __future__ import annotations

from contextcore.permissions import Permissions
from contextcore.security import ServicePermissionInterceptor

# ── RPC → Permission mapping ──────────────────────────────────

RPC_PERMISSION_MAP: dict[str, str] = {
    # Execution
    "ExecuteAgent": Permissions.ROUTER_EXECUTE,
    "StreamAgent": Permissions.ROUTER_EXECUTE,
    "ExecuteDispatcher": Permissions.ROUTER_EXECUTE,
    "StreamDispatcher": Permissions.ROUTER_EXECUTE,
    # Tool Registration (project ↔ router binding)
    "RegisterManifest": Permissions.TOOLS_REGISTER,
    # BiDi Stream (federated tool execution)
    "ToolExecutorStream": Permissions.TOOLS_REGISTER,
}


class RouterPermissionInterceptor(ServicePermissionInterceptor):
    """Router-specific permission interceptor.

    Thin wrapper around ``ServicePermissionInterceptor`` that pre-fills
    the Router RPC permission map and service name.

    Usage::

        interceptor = RouterPermissionInterceptor()
        server = grpc.aio.server(interceptors=[interceptor])
    """

    def __init__(self, *, shield_url: str = "") -> None:
        super().__init__(
            RPC_PERMISSION_MAP,
            service_name="Router",
            shield_url=shield_url,
        )


__all__ = [
    "RouterPermissionInterceptor",
    "RPC_PERMISSION_MAP",
]
