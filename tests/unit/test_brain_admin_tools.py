"""Tenant isolation and permission tests for Brain admin platform tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.permissions import Permissions
from contextunity.core.tokens import TokenBuilder

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.modules.tools.brain_admin_tools import (
    _resolve_tenant,
    get_analytics_summary,
    list_platform_tenants,
    query_traces,
)


def _mint(permissions: list[str], tenants: list[str]):
    return TokenBuilder().mint_root(
        user_ctx={},
        permissions=permissions,
        ttl_s=3600,
        allowed_tenants=tenants,
    )


class TestAdminResolveTenant:
    def test_fail_closed_without_token(self) -> None:
        with (
            patch("contextunity.core.authz.context.get_auth_context", return_value=None),
            patch("contextunity.router.core.context.get_current_access_token", return_value=None),
        ):
            with pytest.raises(SecurityError):
                _resolve_tenant("tenant_a")

    def test_empty_tenant_token_grants_nothing(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.ADMIN_READ], []))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                with pytest.raises(SecurityError, match="grants no tenant access"):
                    _resolve_tenant("tenant_a")
        finally:
            reset_current_access_token(token_ref)

    def test_cross_tenant_refused(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.ADMIN_READ], ["tenant_a"]))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                with pytest.raises(SecurityError, match="not allowed by the caller token"):
                    _resolve_tenant("tenant_b")
        finally:
            reset_current_access_token(token_ref)

    def test_admin_all_omitted_tenant_returns_none(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.ADMIN_ALL], []))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                assert _resolve_tenant(None) is None
        finally:
            reset_current_access_token(token_ref)

    def test_admin_all_may_name_any_tenant(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.ADMIN_ALL], []))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                assert _resolve_tenant("any_tenant") == "any_tenant"
        finally:
            reset_current_access_token(token_ref)


class TestAdminToolPermissions:
    @pytest.mark.asyncio
    async def test_query_traces_refuses_without_admin_read(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.BRAIN_READ], ["tenant_a"]))
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                with pytest.raises(SecurityError, match="admin:read"):
                    await query_traces.ainvoke({"tenant_id": "tenant_a"})
        finally:
            reset_current_access_token(token_ref)

    @pytest.mark.asyncio
    async def test_list_platform_tenants_under_admin_read(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.ADMIN_READ], ["tenant_a"]))
        mock_brain = MagicMock()
        mock_brain.list_tenants = AsyncMock(return_value=[{"id": "tenant_a", "trace_count": 1}])
        try:
            with (
                patch("contextunity.core.authz.context.get_auth_context", return_value=None),
                patch(
                    "contextunity.router.modules.tools.brain_admin_tools._get_brain_client",
                    return_value=mock_brain,
                ),
            ):
                result = await list_platform_tenants.ainvoke({})
        finally:
            reset_current_access_token(token_ref)

        assert result == [{"id": "tenant_a", "trace_count": 1}]
        mock_brain.list_tenants.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_analytics_summary_resolves_tenant(self) -> None:
        token_ref = set_current_access_token(_mint([Permissions.ADMIN_READ], ["tenant_a"]))
        mock_brain = MagicMock()
        mock_brain.get_analytics_summary = AsyncMock(return_value={"total_traces": 3})
        try:
            with (
                patch("contextunity.core.authz.context.get_auth_context", return_value=None),
                patch(
                    "contextunity.router.modules.tools.brain_admin_tools._get_brain_client",
                    return_value=mock_brain,
                ),
            ):
                result = await get_analytics_summary.ainvoke({"tenant_id": None, "hours": 24})
        finally:
            reset_current_access_token(token_ref)

        assert result == {"total_traces": 3}
        mock_brain.get_analytics_summary.assert_awaited_once_with(
            tenant_id="tenant_a",
            hours=24,
        )
