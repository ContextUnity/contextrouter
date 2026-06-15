"""Tenant isolation tests for Redis memory tools.

Pins the contract: the effective tenant comes from the verified caller token,
never from the agent-supplied parameter alone, and resolution fails closed
without a token.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import TokenBuilder

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.modules.tools.redis_memory import _resolve_tenant, store_memory


def _mint(permissions: list[str], tenants: list[str]):
    return TokenBuilder().mint_root(
        user_ctx={},
        permissions=permissions,
        ttl_s=3600,
        allowed_tenants=tenants,
    )


class TestResolveTenant:
    def test_fail_closed_without_token(self):
        with (
            patch("contextunity.core.authz.context.get_auth_context", return_value=None),
            patch("contextunity.router.core.context.get_current_access_token", return_value=None),
        ):
            with pytest.raises(SecurityError):
                _resolve_tenant("tenant_a")

    def test_single_tenant_token_resolves_implicitly(self):
        token_ref = set_current_access_token(_mint(["brain:read"], ["tenant_a"]))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                assert _resolve_tenant(None) == "tenant_a"
        finally:
            reset_current_access_token(token_ref)

    def test_requested_tenant_must_be_allowed(self):
        token_ref = set_current_access_token(_mint(["brain:read"], ["tenant_a"]))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                with pytest.raises(SecurityError, match="not allowed by the caller token"):
                    _resolve_tenant("tenant_b")
        finally:
            reset_current_access_token(token_ref)

    def test_empty_tenant_token_grants_nothing(self):
        token_ref = set_current_access_token(_mint(["brain:read"], []))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                with pytest.raises(SecurityError, match="grants no tenant access"):
                    _resolve_tenant("tenant_a")
        finally:
            reset_current_access_token(token_ref)

    def test_multi_tenant_token_requires_explicit_tenant(self):
        token_ref = set_current_access_token(_mint(["brain:read"], ["tenant_a", "tenant_b"]))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                with pytest.raises(SecurityError, match="tenant_id is required"):
                    _resolve_tenant(None)
                assert _resolve_tenant("tenant_b") == "tenant_b"
        finally:
            reset_current_access_token(token_ref)

    def test_admin_all_token_may_name_any_tenant(self):
        token_ref = set_current_access_token(_mint(["admin:all"], []))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                assert _resolve_tenant("any_tenant") == "any_tenant"
                assert _resolve_tenant(None) == "default"
        finally:
            reset_current_access_token(token_ref)


class TestStoreMemoryToolFailClosed:
    @pytest.mark.asyncio
    async def test_cross_tenant_store_refused(self):
        token_ref = set_current_access_token(_mint(["brain:read"], ["tenant_a"]))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                result = await store_memory.ainvoke(
                    {
                        "key": "k",
                        "value": "v",
                        "session_id": "s1",
                        "tenant_id": "tenant_b",
                    }
                )
        finally:
            reset_current_access_token(token_ref)
        assert result["success"] is False
        assert "not allowed" in str(result.get("error", ""))

    @pytest.mark.asyncio
    async def test_store_without_any_token_refused(self):
        with (
            patch("contextunity.core.authz.context.get_auth_context", return_value=None),
            patch("contextunity.router.core.context.get_current_access_token", return_value=None),
        ):
            result = await store_memory.ainvoke({"key": "k", "value": "v", "session_id": "s1"})
        assert result["success"] is False
