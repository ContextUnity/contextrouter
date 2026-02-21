"""Tests for Router permission interceptor and RPC permission mapping.

Key tests:
- TestRPCPermissionMap: individual method assertions + proto-driven coverage
- TestProtoPermissionCoverage: reads methods directly from the generated pb2_grpc
  descriptor so the test automatically fails when a new RPC is added to the
  proto but not added to RPC_PERMISSION_MAP.
- TestRouterPermissionInterceptor: interceptor construction / mode smoke tests.
"""

from __future__ import annotations

import contextcore.router_pb2 as router_pb2
from contextcore.permissions import Permissions
from contextcore.security import EnforcementMode

from contextrouter.service.interceptors import (
    RPC_PERMISSION_MAP,
    RouterPermissionInterceptor,
)

# ── RPC Permission Map ──────────────────────────────────────────


class TestRPCPermissionMap:
    """Verify all Router RPCs are mapped to correct permissions."""

    def test_execute_agent_mapped(self):
        assert RPC_PERMISSION_MAP["ExecuteAgent"] == Permissions.ROUTER_INVOKE

    def test_stream_agent_mapped(self):
        assert RPC_PERMISSION_MAP["StreamAgent"] == Permissions.ROUTER_INVOKE

    def test_execute_dispatcher_mapped(self):
        assert RPC_PERMISSION_MAP["ExecuteDispatcher"] == Permissions.DISPATCHER_EXECUTE

    def test_stream_dispatcher_mapped(self):
        assert RPC_PERMISSION_MAP["StreamDispatcher"] == Permissions.DISPATCHER_EXECUTE

    def test_register_tools_mapped(self):
        assert RPC_PERMISSION_MAP["RegisterTools"] == Permissions.TOOLS_REGISTER

    def test_deregister_tools_mapped(self):
        assert RPC_PERMISSION_MAP["DeregisterTools"] == Permissions.TOOLS_REGISTER

    def test_tool_executor_stream_mapped(self):
        assert RPC_PERMISSION_MAP["ToolExecutorStream"] == Permissions.TOOLS_REGISTER

    def test_all_rpcs_covered(self):
        """Every Router RPC must be in the map."""
        expected_rpcs = {
            "ExecuteAgent",
            "StreamAgent",
            "ExecuteDispatcher",
            "StreamDispatcher",
            "RegisterTools",
            "DeregisterTools",
            "ToolExecutorStream",
        }
        assert set(RPC_PERMISSION_MAP.keys()) == expected_rpcs

    def test_execution_rpcs_require_invoke_or_execute(self):
        """Execution RPCs should require invoke or execute permissions."""
        exec_rpcs = ["ExecuteAgent", "StreamAgent", "ExecuteDispatcher", "StreamDispatcher"]
        for rpc in exec_rpcs:
            perm = RPC_PERMISSION_MAP[rpc]
            assert ":invoke" in perm or ":execute" in perm, (
                f"{rpc} should require invoke/execute, got {perm}"
            )

    def test_registration_rpcs_require_register(self):
        """Registration RPCs should require register permission."""
        reg_rpcs = ["RegisterTools", "DeregisterTools", "ToolExecutorStream"]
        for rpc in reg_rpcs:
            perm = RPC_PERMISSION_MAP[rpc]
            assert ":register" in perm, f"{rpc} should require register, got {perm}"


# ── Proto-driven coverage ────────────────────────────────────────


def _get_proto_rpc_names() -> set[str]:
    """Extract all RPC method names directly from the RouterService descriptor.

    Uses the compiled pb2 DESCRIPTOR — single source of truth:
    proto → pb2 → RPC_PERMISSION_MAP. No manual list maintenance.
    """
    service = router_pb2.DESCRIPTOR.services_by_name.get("RouterService")
    if service is None:
        raise RuntimeError(
            "RouterService not found in router_pb2.DESCRIPTOR — "
            "did you forget to run compile_protos.sh?"
        )
    return {method.name for method in service.methods}


class TestProtoPermissionCoverage:
    """Proto-driven: every RPC defined in router.proto must be in RPC_PERMISSION_MAP.

    This test reads method names at runtime from the compiled pb2 descriptor,
    so it automatically fails when a new ``rpc`` is added to ``router.proto``
    but the developer forgets to add it to ``RPC_PERMISSION_MAP``.
    """

    def test_every_proto_method_in_map(self):
        """No orphan RPC: each proto method must have a required permission."""
        proto_methods = _get_proto_rpc_names()
        missing = proto_methods - set(RPC_PERMISSION_MAP.keys())
        assert not missing, (
            f"These RouterService RPCs are defined in router.proto but missing "
            f"from RPC_PERMISSION_MAP — add them:\n  {sorted(missing)}"
        )

    def test_no_phantom_entries_in_map(self):
        """No phantom entries: every map key must correspond to an actual proto RPC."""
        proto_methods = _get_proto_rpc_names()
        phantom = set(RPC_PERMISSION_MAP.keys()) - proto_methods
        assert not phantom, (
            f"These entries in RPC_PERMISSION_MAP have no matching RPC in router.proto — "
            f"stale entries, remove them:\n  {sorted(phantom)}"
        )

    def test_map_and_proto_are_in_sync(self):
        """Full symmetry: RPC_PERMISSION_MAP covers exactly the proto methods."""
        proto_methods = _get_proto_rpc_names()
        assert set(RPC_PERMISSION_MAP.keys()) == proto_methods, (
            f"RPC_PERMISSION_MAP out of sync with router.proto.\n"
            f"  In proto, not in map : {proto_methods - set(RPC_PERMISSION_MAP.keys())}\n"
            f"  In map, not in proto : {set(RPC_PERMISSION_MAP.keys()) - proto_methods}"
        )


# ── Interceptor ─────────────────────────────────────────────────


class TestRouterPermissionInterceptor:
    """Test interceptor construction."""

    def test_warn_by_default(self):
        interceptor = RouterPermissionInterceptor()
        assert interceptor._mode == EnforcementMode.WARN

    def test_enforce_mode(self):
        interceptor = RouterPermissionInterceptor(enforcement=EnforcementMode.ENFORCE)
        assert interceptor._mode == EnforcementMode.ENFORCE

    def test_warn_mode(self):
        interceptor = RouterPermissionInterceptor(enforcement=EnforcementMode.WARN)
        assert interceptor._mode == EnforcementMode.WARN

    def test_service_name(self):
        interceptor = RouterPermissionInterceptor()
        assert interceptor._service_name == "Router"

    def test_rpc_map_set(self):
        interceptor = RouterPermissionInterceptor()
        assert interceptor._rpc_map == RPC_PERMISSION_MAP
