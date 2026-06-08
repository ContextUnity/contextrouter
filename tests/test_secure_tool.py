import time

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import ContextToken
from langchain_core.tools import BaseTool, tool

from contextunity.router.modules.tools.secure import SecureTool

# ─── Fixtures ───


def _make_token(
    permissions: tuple[str, ...] = (),
    allowed_tenants: tuple[str, ...] = ("default",),
) -> ContextToken:
    """Create a real ContextToken for testing."""
    return ContextToken(
        token_id="test-token",
        user_id="test-user",
        permissions=permissions,
        allowed_tenants=allowed_tenants,
        exp_unix=time.time() + 300,
    )


class RawDummyTool(BaseTool):
    """A raw BaseTool without any security enforcement."""

    name: str = "raw_dummy"
    description: str = "A raw tool for testing"

    def _run(self, query: str = "") -> str:
        return f"raw_result:{query}"

    async def _arun(self, query: str = "") -> str:
        return f"async_raw_result:{query}"


class DirectSecureTool(SecureTool):
    """A SecureTool subclass (no wrapping, direct implementation)."""

    name: str = "direct_secure"
    description: str = "A directly implemented secure tool"
    required_permission: str = "tool:direct_secure"

    def _run(self, query: str = "") -> str:
        self._enforce_permission()
        return f"direct_result:{query}"

    async def _arun(self, query: str = "") -> str:
        self._enforce_permission()
        return f"async_direct_result:{query}"


@pytest.fixture
def raw_tool():
    return RawDummyTool()


@pytest.fixture
def secure_wrapped(raw_tool):
    return SecureTool.wrap(raw_tool, permission="tool:raw_dummy")


def _set_token(token):
    """Helper: set access token in runtime context."""
    from contextunity.router.core.context import set_current_access_token

    return set_current_access_token(token)


def _reset_token(ref):
    """Helper: reset access token."""
    from contextunity.router.core.context import reset_current_access_token

    reset_current_access_token(ref)


# ─── Test: SecureTool.wrap() ───


class TestWrap:
    def test_wrap_raw_basetool(self, raw_tool):
        secure = SecureTool.wrap(raw_tool)
        assert isinstance(secure, SecureTool)
        assert secure.name == "raw_dummy"
        assert secure.required_permission == "tool:raw_dummy"
        assert secure.required_scope == "read"
        assert secure.wrapped_tool is raw_tool

    def test_wrap_preserves_description(self, raw_tool):
        secure = SecureTool.wrap(raw_tool)
        assert secure.description == "A raw tool for testing"

    def test_wrap_custom_permission(self, raw_tool):
        secure = SecureTool.wrap(raw_tool, permission="tool:custom")
        assert secure.required_permission == "tool:custom"

    def test_wrap_idempotent(self, secure_wrapped):
        """Wrapping a SecureTool returns the same instance."""
        double_wrapped = SecureTool.wrap(secure_wrapped)
        assert double_wrapped is secure_wrapped


# ─── Test: Permission Enforcement ───


class TestPermissionEnforcement:
    def test_no_token_raises(self, secure_wrapped):
        """No access token → PermissionError (fail-closed)."""
        ref = _set_token(None)
        try:
            with pytest.raises(SecurityError, match="No access token"):
                secure_wrapped._run(query="test")
        finally:
            _reset_token(ref)

    def test_wrong_permission_raises(self, secure_wrapped):
        """Token without matching permission → PermissionError."""
        token = _make_token(permissions=("tool:other_tool",))
        ref = _set_token(token)
        try:
            with pytest.raises(SecurityError, match="Permission denied"):
                secure_wrapped._run(query="test")
        finally:
            _reset_token(ref)

    def test_correct_permission_executes(self, secure_wrapped):
        """Token with matching permission → tool executes."""
        token = _make_token(permissions=("tool:raw_dummy",))
        ref = _set_token(token)
        try:
            result = secure_wrapped._run(query="hello")
            assert result == "raw_result:hello"
        finally:
            _reset_token(ref)

    def test_wildcard_permission_executes(self, secure_wrapped):
        """Token with tool:* wildcard → tool executes."""
        token = _make_token(permissions=("tool:*",))
        ref = _set_token(token)
        try:
            result = secure_wrapped._run(query="wildcard")
            assert result == "raw_result:wildcard"
        finally:
            _reset_token(ref)

    def test_admin_all_permission_executes(self, secure_wrapped):
        """Token with admin:all → tool executes."""
        token = _make_token(permissions=("admin:all",))
        ref = _set_token(token)
        try:
            result = secure_wrapped._run(query="admin")
            assert result == "raw_result:admin"
        finally:
            _reset_token(ref)

    def test_tool_execution_logs_provenance(self, secure_wrapped):
        """Execution appends flat provenance trail (prefix:name:mode) via runtime_context."""
        from contextunity.router.core.context import (
            get_accumulated_provenance,
            init_provenance_accumulator,
            reset_provenance_accumulator,
        )

        token = _make_token(
            permissions=(
                "tool:raw_dummy",
                "tool:raw_dummy:read",
            )
        )
        ref = _set_token(token)
        # We must initialize the accumulator context to capture provenance
        accum_ref = init_provenance_accumulator()

        try:
            secure_wrapped._run(query="provenance")
            history = get_accumulated_provenance()
            assert len(history) == 1
            assert history[0] == "tool:raw_dummy:read"  # extracted read scope from token
        finally:
            reset_provenance_accumulator(accum_ref)
            _reset_token(ref)

    def test_tool_execution_logs_federated_provenance(self, raw_tool):
        """Execution appends federated provenance if federated tag is present."""
        from contextunity.router.core.context import (
            get_accumulated_provenance,
            init_provenance_accumulator,
            reset_provenance_accumulator,
        )

        raw_tool.tags = ["federated"]
        federated_secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy")

        token = _make_token(permissions=("tool:raw_dummy", "tool:raw_dummy:execute"))
        ref = _set_token(token)
        accum_ref = init_provenance_accumulator()

        try:
            federated_secure._run(query="provenance")
            history = get_accumulated_provenance()
            assert "federated_tool:raw_dummy:execute" in history
        finally:
            reset_provenance_accumulator(accum_ref)
            _reset_token(ref)


# ─── Test: Async Enforcement ───


class TestAsyncEnforcement:
    @pytest.mark.asyncio
    async def test_async_no_token_raises(self, secure_wrapped):
        ref = _set_token(None)
        try:
            with pytest.raises(SecurityError, match="No access token"):
                await secure_wrapped._arun(query="test")
        finally:
            _reset_token(ref)

    @pytest.mark.asyncio
    async def test_async_correct_permission_executes(self, secure_wrapped):
        token = _make_token(permissions=("tool:raw_dummy",))
        ref = _set_token(token)
        try:
            result = await secure_wrapped._arun(query="async_hello")
            assert result == "async_raw_result:async_hello"
        finally:
            _reset_token(ref)


# ─── Test: mark_infra() ───


class TestMarkInfra:
    def test_infra_skips_auth_no_token(self):
        """mark_infra() tools execute even without a token."""
        raw = RawDummyTool(name="log_execution_trace", description="trace tool")
        secure = SecureTool.mark_infra(raw)
        assert secure.skip_auth is True
        ref = _set_token(None)
        try:
            result = secure._run(query="trace")
            assert result == "raw_result:trace"
        finally:
            _reset_token(ref)

    def test_infra_on_existing_securetool(self):
        """mark_infra() can be applied to an existing SecureTool."""
        st = SecureTool(name="cleanup", description="cleanup tool")
        assert st.skip_auth is False
        marked = SecureTool.mark_infra(st)
        assert marked is st  # same instance
        assert marked.skip_auth is True

    def test_name_spoofing_blocked(self):
        """Tool named 'log_execution_trace' does NOT skip auth
        unless explicitly mark_infra'd."""
        fake = RawDummyTool(name="log_execution_trace", description="malicious tool")
        secure = SecureTool.wrap(fake)
        assert secure.skip_auth is False  # NOT infra
        ref = _set_token(None)
        try:
            with pytest.raises(SecurityError, match="No access token"):
                secure._run(query="spoofed")
        finally:
            _reset_token(ref)


# ─── Test: Direct Subclass ───


class TestDirectSubclass:
    def test_direct_subclass_enforces(self):
        """SecureTool subclass with _enforce_permission() in _run() works."""
        tool_inst = DirectSecureTool()
        token = _make_token(permissions=("tool:direct_secure",))
        ref = _set_token(token)
        try:
            result = tool_inst._run(query="direct")
            assert result == "direct_result:direct"
        finally:
            _reset_token(ref)

    def test_direct_subclass_blocks_without_token(self):
        tool_inst = DirectSecureTool()
        ref = _set_token(None)
        try:
            with pytest.raises(SecurityError):
                tool_inst._run(query="blocked")
        finally:
            _reset_token(ref)


# ─── Test: register_tool() auto-wrapping ───


class TestRegisterToolGate:
    def test_register_raw_tool_wraps(self, raw_tool):
        """register_tool() auto-wraps raw BaseTool → SecureTool."""
        from contextunity.router.modules.tools import _tool_registry, register_tool

        original_registry = _tool_registry.copy()
        try:
            register_tool(raw_tool, permission="tool:raw_dummy")
            registered = _tool_registry["raw_dummy"]
            assert isinstance(registered, SecureTool)
            assert registered.required_permission == "tool:raw_dummy"
            assert registered.skip_auth is False  # never infra from register
        finally:
            _tool_registry.clear()
            _tool_registry.update(original_registry)

    def test_register_secure_tool_keeps(self, secure_wrapped):
        """register_tool() keeps SecureTool as-is."""
        from contextunity.router.modules.tools import _tool_registry, register_tool

        original_registry = _tool_registry.copy()
        try:
            register_tool(secure_wrapped)
            registered = _tool_registry["raw_dummy"]
            assert isinstance(registered, SecureTool)
            assert registered is secure_wrapped
        finally:
            _tool_registry.clear()
            _tool_registry.update(original_registry)

    def test_register_with_decorator(self):
        """@tool decorated function gets wrapped."""
        from contextunity.router.modules.tools import _tool_registry, register_tool

        @tool
        def my_test_tool(query: str) -> str:
            """A test tool."""
            return f"result:{query}"

        original_registry = _tool_registry.copy()
        try:
            register_tool(my_test_tool)
            registered = _tool_registry["my_test_tool"]
            assert isinstance(registered, SecureTool)
            assert registered.required_permission == "tool:my_test_tool"
            assert registered.skip_auth is False
        finally:
            _tool_registry.clear()
            _tool_registry.update(original_registry)

    def test_register_never_grants_infra(self):
        """register_tool() never sets skip_auth even for infra-named tools."""
        from contextunity.router.modules.tools import _tool_registry, register_tool

        fake = RawDummyTool(name="log_execution_trace", description="malicious")
        original_registry = _tool_registry.copy()
        try:
            register_tool(fake)
            registered = _tool_registry["log_execution_trace"]
            assert isinstance(registered, SecureTool)
            assert registered.skip_auth is False  # NOT infra
        finally:
            _tool_registry.clear()
            _tool_registry.update(original_registry)


class TestTenantIsolation:
    """Verify that tools bound to a tenant reject tokens from other tenants."""

    def test_cross_tenant_blocked(self, raw_tool):
        """Token for tenant 'tenant_y' cannot execute tool bound to 'tenant_a'."""
        secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy", tenant="tenant_a")
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("tenant_y",),
        )
        ref = _set_token(token)
        try:
            with pytest.raises(SecurityError, match="Tenant isolation"):
                secure._run(query="cross-tenant")
        finally:
            _reset_token(ref)

    def test_same_tenant_allowed(self, raw_tool):
        """Token for tenant 'tenant_a' CAN execute tool bound to 'tenant_a'."""
        secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy", tenant="tenant_a")
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("tenant_a",),
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="same-tenant")
            assert result == "raw_result:same-tenant"
        finally:
            _reset_token(ref)

    def test_multi_tenant_token_allowed(self, raw_tool):
        """Token with multiple tenants including 'tenant_a' can access 'tenant_a' tools."""
        secure = SecureTool.wrap(raw_tool, tenant="tenant_a")
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("tenant_x", "tenant_a", "tenant_z"),
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="multi-tenant")
            assert result == "raw_result:multi-tenant"
        finally:
            _reset_token(ref)

    def test_no_binding_allows_any_tenant(self, raw_tool):
        """Tool with no bound_tenant allows execution from any tenant."""
        secure = SecureTool.wrap(raw_tool)  # no tenant
        assert secure.bound_tenant == ""
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("any_project",),
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="global")
            assert result == "raw_result:global"
        finally:
            _reset_token(ref)

    def test_wrap_propagates_tenant(self, raw_tool):
        """wrap() stores tenant in bound_tenant."""
        secure = SecureTool.wrap(raw_tool, tenant="tenant_a")
        assert secure.bound_tenant == "tenant_a"

    @pytest.mark.asyncio
    async def test_async_cross_tenant_blocked(self, raw_tool):
        """Async execution also enforces tenant."""
        secure = SecureTool.wrap(raw_tool, tenant="tenant_a")
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("other_project",),
        )
        ref = _set_token(token)
        try:
            with pytest.raises(SecurityError, match="Tenant isolation"):
                await secure._arun(query="async-cross")
        finally:
            _reset_token(ref)

    def test_infra_tool_skips_tenant_check(self, raw_tool):
        """mark_infra() tools skip tenant check even if bound."""
        secure = SecureTool.wrap(raw_tool, tenant="tenant_a")
        secure = SecureTool.mark_infra(secure)
        # No token at all — should still execute (infra skips everything)
        ref = _set_token(None)
        try:
            result = secure._run(query="infra")
            assert result == "raw_result:infra"
        finally:
            _reset_token(ref)

    def test_admin_all_bypasses_tenant_isolation(self, raw_tool):
        """admin:all permission allows access to any tenant-bound tool."""
        secure = SecureTool.wrap(raw_tool, tenant="tenant_a")
        token = _make_token(
            permissions=("admin:all",),
            allowed_tenants=(),  # empty — admin sees all
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="admin")
            assert result == "raw_result:admin"
        finally:
            _reset_token(ref)

    def test_admin_all_with_specific_tenant_also_works(self, raw_tool):
        """admin:all with explicit tenants still works."""
        secure = SecureTool.wrap(raw_tool, tenant="tenant_a")
        token = _make_token(
            permissions=("admin:all",),
            allowed_tenants=("other",),  # doesn't include tenant_a, but admin:all bypasses
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="admin-specific")
            assert result == "raw_result:admin-specific"
        finally:
            _reset_token(ref)

    def test_import_error_fails_closed(self, raw_tool):
        """If contextunity.core.authz can't be imported, tool is BLOCKED."""
        secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy")
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("default",),
        )
        ref = _set_token(token)
        try:
            import sys

            # Temporarily hide contextunity.core.authz
            original = sys.modules.get("contextunity.core.authz")
            sys.modules["contextunity.core.authz"] = None  # type: ignore
            try:
                with pytest.raises((SecurityError, ImportError, TypeError)):
                    secure._run(query="should-fail")
            finally:
                if original is not None:
                    sys.modules["contextunity.core.authz"] = original
                else:
                    sys.modules.pop("contextunity.core.authz", None)
        finally:
            _reset_token(ref)


# ─── Test: Authoritative Context Injection (anti-forgery) ───


class TestAuthoritativeContextInjection:
    """Verify _inject_authoritative_context overwrites LLM-controllable kwargs."""

    def _make_tool_with_schema(self, field_names: list[str]):
        """Create a wrapped SecureTool whose args_schema exposes given fields."""
        from pydantic import BaseModel

        annotations = {n: str for n in field_names}
        defaults = {n: "" for n in field_names}
        schema = type("DynSchema", (BaseModel,), {"__annotations__": annotations, **defaults})

        raw = RawDummyTool()
        raw.args_schema = schema
        return SecureTool.wrap(raw, permission="tool:raw_dummy")

    def test_tenant_id_overwritten_from_token(self):
        """LLM-injected tenant_id is replaced with token's allowed_tenants[0]."""
        secure = self._make_tool_with_schema(["tenant_id", "query"])
        token = _make_token(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("real_tenant",),
        )
        ref = _set_token(token)
        try:
            result = secure._inject_authoritative_context(
                {"tenant_id": "spoofed_tenant", "query": "hello"}
            )
            assert result["tenant_id"] == "real_tenant"
            assert result["query"] == "hello"
        finally:
            _reset_token(ref)

    def test_user_id_overwritten_from_token(self):
        """LLM-injected user_id is replaced with token's user_id."""
        secure = self._make_tool_with_schema(["user_id", "query"])
        token = _make_token(permissions=("tool:raw_dummy",))
        ref = _set_token(token)
        try:
            result = secure._inject_authoritative_context(
                {"user_id": "spoofed_user", "query": "hello"}
            )
            assert result["user_id"] == "test-user"
        finally:
            _reset_token(ref)

    def test_infra_tool_skips_injection(self):
        """skip_auth tools return kwargs unchanged."""
        raw = RawDummyTool()
        secure = SecureTool.mark_infra(raw)
        result = secure._inject_authoritative_context(
            {"tenant_id": "anything", "user_id": "anything"}
        )
        assert result["tenant_id"] == "anything"
        assert result["user_id"] == "anything"

    def test_no_token_returns_kwargs_unchanged(self):
        """Without token, kwargs pass through unchanged."""
        raw = RawDummyTool()
        secure = SecureTool.wrap(raw)
        ref = _set_token(None)
        try:
            result = secure._inject_authoritative_context({"tenant_id": "x", "user_id": "y"})
            assert result["tenant_id"] == "x"
        finally:
            _reset_token(ref)


# ─── Test: Provenance Prefix Resolution ───


class TestProvenancePrefix:
    def test_federated_tool_prefix(self, raw_tool):
        raw_tool.tags = ["federated"]
        secure = SecureTool.wrap(raw_tool)
        assert secure._resolve_provenance_prefix() == "federated_tool"

    def test_privacy_tool_prefix(self):
        secure = SecureTool(
            name="anonymize", description="anon", required_permission="privacy:anonymize"
        )
        assert secure._resolve_provenance_prefix() == "privacy_tool"

    def test_shield_tool_prefix(self):
        secure = SecureTool(name="scan", description="scan", required_permission="shield:scan")
        assert secure._resolve_provenance_prefix() == "shield_tool"

    def test_regular_tool_prefix(self, raw_tool):
        secure = SecureTool.wrap(raw_tool)
        assert secure._resolve_provenance_prefix() == "tool"


# ─── Test: NotImplementedError path ───


class TestNoWrappedTool:
    def test_bare_securetool_run_raises(self):
        """SecureTool without wrapped_tool and without _run override raises."""
        st = SecureTool(name="bare", description="bare", skip_auth=True)
        with pytest.raises(NotImplementedError):
            st._run(query="test")

    @pytest.mark.asyncio
    async def test_bare_securetool_arun_raises(self):
        st = SecureTool(name="bare", description="bare", skip_auth=True)
        with pytest.raises(NotImplementedError):
            await st._arun(query="test")
