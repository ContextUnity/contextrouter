"""gRPC interceptor for ContextRouter permission enforcement.

Maps each Router RPC method to the exact permission required
and validates the ContextToken carries that permission.

Delegates to ``contextcore.security.ServicePermissionInterceptor``
for unified enforcement logic. Router only owns the RPC_PERMISSION_MAP.
"""

from __future__ import annotations

from contextcore.permissions import Permissions
from contextcore.security import EnforcementMode, ServicePermissionInterceptor

# ── RPC → Permission mapping ──────────────────────────────────

RPC_PERMISSION_MAP: dict[str, str] = {
    # Execution
    "ExecuteAgent": Permissions.ROUTER_INVOKE,
    "StreamAgent": Permissions.ROUTER_INVOKE,
    "ExecuteDispatcher": Permissions.DISPATCHER_EXECUTE,
    "StreamDispatcher": Permissions.DISPATCHER_EXECUTE,
    # Tool Registration (project ↔ router binding)
    "RegisterTools": Permissions.TOOLS_REGISTER,
    "DeregisterTools": Permissions.TOOLS_REGISTER,
    # BiDi Stream (federated tool execution)
    "ToolExecutorStream": Permissions.TOOLS_REGISTER,
}


class RouterPermissionInterceptor(ServicePermissionInterceptor):
    """Router-specific permission interceptor.

    Thin wrapper around ``ServicePermissionInterceptor`` that pre-fills
    the Router RPC permission map and service name.

    Usage::

        interceptor = RouterPermissionInterceptor(enforcement=EnforcementMode.WARN)
        server = grpc.aio.server(interceptors=[interceptor])
    """

    def __init__(self, *, enforcement: EnforcementMode | None = None) -> None:
        super().__init__(
            RPC_PERMISSION_MAP,
            service_name="Router",
            enforcement=enforcement,
        )


__all__ = [
    "RouterPermissionInterceptor",
    "RPC_PERMISSION_MAP",
]
