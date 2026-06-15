"""Tests for Router→Shield SPOT metadata (no local mint_service_token)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import TokenBuilder

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.service.shield_client import shield_metadata


class TestShieldMetadataSpot:
    def test_forwards_auth_context_token_string(self):
        with patch(
            "contextunity.core.authz.context.get_auth_context",
            return_value=SimpleNamespace(token_string="project:hmac-001.payload.sig"),
        ):
            metadata = shield_metadata()
        assert metadata == (("authorization", "Bearer project:hmac-001.payload.sig"),)

    def test_serializes_graph_execution_token_when_no_auth_string(self):
        token = TokenBuilder().mint_root(
            user_ctx={},
            permissions=["shield:secrets:read"],
            ttl_s=3600,
            allowed_tenants=["default"],
        )
        token_ref = set_current_access_token(token)
        try:
            with (
                patch("contextunity.core.authz.context.get_auth_context", return_value=None),
                patch(
                    "contextunity.core.token_utils.serialize_token",
                    return_value="attenuated.wire.token",
                ) as mock_serialize,
            ):
                metadata = shield_metadata()
            mock_serialize.assert_called_once()
            assert metadata == (("authorization", "Bearer attenuated.wire.token"),)
        finally:
            reset_current_access_token(token_ref)

    def test_fail_closed_without_caller_token(self):
        with (
            patch("contextunity.core.authz.context.get_auth_context", return_value=None),
            patch(
                "contextunity.router.core.context.get_current_access_token",
                return_value=None,
            ),
        ):
            with pytest.raises(SecurityError, match="background service-token minting"):
                shield_metadata()

    def test_shield_client_module_has_no_mint_service_token_call(self):
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "contextunity"
            / "router"
            / "service"
            / "shield_client.py"
        ).read_text(encoding="utf-8")
        assert re.search(r"\bmint_service_token\s*\(", source) is None
        assert (
            "from contextunity.core.tokens import" not in source
            or "mint_service_token" not in source
        )
