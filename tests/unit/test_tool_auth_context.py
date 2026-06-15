"""Tests for resolve_tool_context_token (Brain tools fail-closed)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import ContextToken, TokenBuilder

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.modules.tools.auth_context import resolve_tool_context_token


class TestResolveToolContextToken:
    def test_prefers_auth_context_token(self):
        expected = ContextToken(token_id="grpc-token", permissions=("brain:read",))
        with patch(
            "contextunity.core.authz.context.get_auth_context",
            return_value=SimpleNamespace(token=expected),
        ):
            assert resolve_tool_context_token() is expected

    def test_falls_back_to_graph_contextvar(self):
        expected = TokenBuilder().mint_root(
            user_ctx={},
            permissions=["brain:read"],
            ttl_s=3600,
            allowed_tenants=["default"],
        )
        token_ref = set_current_access_token(expected)
        try:
            with patch("contextunity.core.authz.context.get_auth_context", return_value=None):
                assert resolve_tool_context_token() is expected
        finally:
            reset_current_access_token(token_ref)

    def test_fail_closed_without_token(self):
        with (
            patch("contextunity.core.authz.context.get_auth_context", return_value=None),
            patch(
                "contextunity.router.core.context.get_current_access_token",
                return_value=None,
            ),
        ):
            with pytest.raises(SecurityError, match="Brain tool requires"):
                resolve_tool_context_token()
