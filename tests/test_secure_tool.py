"""Tests for SecureTool — mandatory permission enforcement on every tool call."""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool, tool

from contextrouter.modules.tools.secure import SecureTool

# ─── Fixtures ───


class DummyToken:
    """Fake ContextToken for testing."""

    def __init__(
        self,
        permissions: tuple[str, ...] = (),
        allowed_tenants: tuple[str, ...] = ("default",),
    ):
        self.token_id = "test-token"
        self.user_id = "test-user"
        self.agent_id = ""
        self.user_namespace = "default"
        self.permissions = permissions
        self.allowed_tenants = allowed_tenants


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
    from contextrouter.cortex.runtime_context import set_current_access_token

    return set_current_access_token(token)


def _reset_token(ref):
    """Helper: reset access token."""
    from contextrouter.cortex.runtime_context import reset_current_access_token

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

    def test_wrap_custom_scope(self, raw_tool):
        secure = SecureTool.wrap(raw_tool, scope="write")
        assert secure.required_scope == "write"

    def test_wrap_idempotent(self, secure_wrapped):
        """Wrapping a SecureTool returns the same instance."""
        double_wrapped = SecureTool.wrap(secure_wrapped)
        assert double_wrapped is secure_wrapped

    def test_wrap_sets_permission_on_empty_securetool(self):
        """Wrapping SecureTool with empty permission updates it."""
        st = SecureTool(name="test", description="test")
        assert st.required_permission == ""
        result = SecureTool.wrap(st, permission="tool:test")
        assert result.required_permission == "tool:test"
        assert result is st  # same instance

    def test_wrap_does_not_override_existing_permission(self):
        """Wrapping SecureTool with existing permission doesn't override."""
        st = SecureTool(name="test", description="test", required_permission="tool:original")
        result = SecureTool.wrap(st, permission="tool:new")
        assert result.required_permission == "tool:original"

    def test_wrap_never_sets_skip_auth(self, raw_tool):
        """wrap() NEVER sets skip_auth — even if tool name matches old whitelist."""
        fake_infra = RawDummyTool(name="log_execution_trace", description="fake infra")
        secure = SecureTool.wrap(fake_infra)
        assert secure.skip_auth is False


# ─── Test: Permission Enforcement ───


class TestPermissionEnforcement:
    def test_no_token_raises(self, secure_wrapped):
        """No access token → PermissionError (fail-closed)."""
        ref = _set_token(None)
        try:
            with pytest.raises(PermissionError, match="No access token"):
                secure_wrapped._run(query="test")
        finally:
            _reset_token(ref)

    def test_wrong_permission_raises(self, secure_wrapped):
        """Token without matching permission → PermissionError."""
        token = DummyToken(permissions=("tool:other_tool",))
        ref = _set_token(token)
        try:
            with pytest.raises(PermissionError, match="Permission denied"):
                secure_wrapped._run(query="test")
        finally:
            _reset_token(ref)

    def test_correct_permission_executes(self, secure_wrapped):
        """Token with matching permission → tool executes."""
        token = DummyToken(permissions=("tool:raw_dummy",))
        ref = _set_token(token)
        try:
            result = secure_wrapped._run(query="hello")
            assert result == "raw_result:hello"
        finally:
            _reset_token(ref)

    def test_wildcard_permission_executes(self, secure_wrapped):
        """Token with tool:* wildcard → tool executes."""
        token = DummyToken(permissions=("tool:*",))
        ref = _set_token(token)
        try:
            result = secure_wrapped._run(query="wildcard")
            assert result == "raw_result:wildcard"
        finally:
            _reset_token(ref)

    def test_admin_all_permission_executes(self, secure_wrapped):
        """Token with admin:all → tool executes."""
        token = DummyToken(permissions=("admin:all",))
        ref = _set_token(token)
        try:
            result = secure_wrapped._run(query="admin")
            assert result == "raw_result:admin"
        finally:
            _reset_token(ref)


# ─── Test: Async Enforcement ───


class TestAsyncEnforcement:
    @pytest.mark.asyncio
    async def test_async_no_token_raises(self, secure_wrapped):
        ref = _set_token(None)
        try:
            with pytest.raises(PermissionError, match="No access token"):
                await secure_wrapped._arun(query="test")
        finally:
            _reset_token(ref)

    @pytest.mark.asyncio
    async def test_async_correct_permission_executes(self, secure_wrapped):
        token = DummyToken(permissions=("tool:raw_dummy",))
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
            with pytest.raises(PermissionError, match="No access token"):
                secure._run(query="spoofed")
        finally:
            _reset_token(ref)


# ─── Test: Direct Subclass ───


class TestDirectSubclass:
    def test_direct_subclass_enforces(self):
        """SecureTool subclass with _enforce_permission() in _run() works."""
        tool_inst = DirectSecureTool()
        token = DummyToken(permissions=("tool:direct_secure",))
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
            with pytest.raises(PermissionError):
                tool_inst._run(query="blocked")
        finally:
            _reset_token(ref)


# ─── Test: register_tool() auto-wrapping ───


class TestRegisterToolGate:
    def test_register_raw_tool_wraps(self, raw_tool):
        """register_tool() auto-wraps raw BaseTool → SecureTool."""
        from contextrouter.modules.tools import _tool_registry, register_tool

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
        from contextrouter.modules.tools import _tool_registry, register_tool

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
        from contextrouter.modules.tools import _tool_registry, register_tool

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
        from contextrouter.modules.tools import _tool_registry, register_tool

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


# ─── Test: repr ───


class TestRepr:
    def test_repr_wrapped(self, secure_wrapped):
        r = repr(secure_wrapped)
        assert "raw_dummy" in r
        assert "tool:raw_dummy" in r
        assert "wrapped=" in r

    def test_repr_direct(self):
        st = SecureTool(name="test", description="test", required_permission="tool:test")
        r = repr(st)
        assert "test" in r
        assert "wrapped=" not in r

    def test_repr_infra(self):
        raw = RawDummyTool(name="trace", description="trace")
        secure = SecureTool.mark_infra(raw)
        r = repr(secure)
        assert "infra=True" in r


# ─── Test: effective_permission fallback ───


class TestEffectivePermission:
    def test_explicit_permission(self):
        st = SecureTool(name="foo", description="", required_permission="tool:custom")
        assert st._effective_permission() == "tool:custom"

    def test_default_permission(self):
        st = SecureTool(name="foo", description="")
        assert st._effective_permission() == "tool:foo"


# ─── Test: Tenant Isolation (VULN-1 fix) ───


class TestTenantIsolation:
    """Verify that tools bound to a tenant reject tokens from other tenants."""

    def test_cross_tenant_blocked(self, raw_tool):
        """Token for tenant 'hospital_B' cannot execute tool bound to 'nszu'."""
        secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy", tenant="nszu")
        token = DummyToken(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("hospital_B",),
        )
        ref = _set_token(token)
        try:
            with pytest.raises(PermissionError, match="Tenant isolation"):
                secure._run(query="cross-tenant")
        finally:
            _reset_token(ref)

    def test_same_tenant_allowed(self, raw_tool):
        """Token for tenant 'nszu' CAN execute tool bound to 'nszu'."""
        secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy", tenant="nszu")
        token = DummyToken(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("nszu",),
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="same-tenant")
            assert result == "raw_result:same-tenant"
        finally:
            _reset_token(ref)

    def test_multi_tenant_token_allowed(self, raw_tool):
        """Token with multiple tenants including 'nszu' can access 'nszu' tools."""
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
        token = DummyToken(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("hospital_A", "nszu", "hospital_C"),
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
        token = DummyToken(
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
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
        assert secure.bound_tenant == "nszu"

    def test_wrap_idempotent_sets_tenant(self):
        """Wrapping a SecureTool without tenant sets it if provided."""
        st = SecureTool(name="test", description="test")
        assert st.bound_tenant == ""
        SecureTool.wrap(st, tenant="nszu")
        assert st.bound_tenant == "nszu"

    def test_wrap_does_not_override_tenant(self):
        """Wrapping a SecureTool with existing tenant doesn't override."""
        st = SecureTool(name="test", description="test", bound_tenant="nszu")
        SecureTool.wrap(st, tenant="other")
        assert st.bound_tenant == "nszu"  # preserved

    def test_register_tool_with_tenant(self, raw_tool):
        """register_tool(tenant=...) binds tenant to the tool."""
        from contextrouter.modules.tools import _tool_registry, register_tool

        original_registry = _tool_registry.copy()
        try:
            register_tool(raw_tool, permission="tool:raw_dummy", tenant="nszu")
            registered = _tool_registry["raw_dummy"]
            assert isinstance(registered, SecureTool)
            assert registered.bound_tenant == "nszu"
        finally:
            _tool_registry.clear()
            _tool_registry.update(original_registry)

    def test_tenant_in_repr(self, raw_tool):
        """Bound tenant appears in repr."""
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
        r = repr(secure)
        assert "tenant='nszu'" in r

    @pytest.mark.asyncio
    async def test_async_cross_tenant_blocked(self, raw_tool):
        """Async execution also enforces tenant."""
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
        token = DummyToken(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("other_project",),
        )
        ref = _set_token(token)
        try:
            with pytest.raises(PermissionError, match="Tenant isolation"):
                await secure._arun(query="async-cross")
        finally:
            _reset_token(ref)

    def test_infra_tool_skips_tenant_check(self, raw_tool):
        """mark_infra() tools skip tenant check even if bound."""
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
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
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
        token = DummyToken(
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
        secure = SecureTool.wrap(raw_tool, tenant="nszu")
        token = DummyToken(
            permissions=("admin:all",),
            allowed_tenants=("other",),  # doesn't include nszu, but admin:all bypasses
        )
        ref = _set_token(token)
        try:
            result = secure._run(query="admin-specific")
            assert result == "raw_result:admin-specific"
        finally:
            _reset_token(ref)

    def test_import_error_fails_closed(self, raw_tool):
        """If contextcore.permissions can't be imported, tool is BLOCKED."""
        secure = SecureTool.wrap(raw_tool, permission="tool:raw_dummy")
        token = DummyToken(
            permissions=("tool:raw_dummy",),
            allowed_tenants=("default",),
        )
        ref = _set_token(token)
        try:
            import sys

            # Temporarily hide contextcore.permissions
            original = sys.modules.get("contextcore.permissions")
            sys.modules["contextcore.permissions"] = None  # type: ignore
            try:
                with pytest.raises(PermissionError, match="fail-closed"):
                    secure._run(query="should-fail")
            finally:
                if original is not None:
                    sys.modules["contextcore.permissions"] = original
                else:
                    sys.modules.pop("contextcore.permissions", None)
        finally:
            _reset_token(ref)
