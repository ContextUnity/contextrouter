"""Behavioral tests for registration_auth token extraction and verification.

Tests cover:
  - Bearer token extraction from gRPC metadata
  - Missing/empty authorization header handling
  - Non-Bearer authorization header handling
  - Token structure validation (composite kid wire format)
  - Project ID mismatch detection
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from contextunity.core.exceptions import SecurityError

from contextunity.router.service.registration_auth import (
    build_registration_verifier,
    extract_registration_token_string,
)

# ── Fake gRPC Context ────────────────────────────────────────────────────


def _make_grpc_context(metadata: dict | None = None) -> SimpleNamespace:
    """Create a fake gRPC context with configurable metadata."""
    meta_list = list((metadata or {}).items())
    ctx = SimpleNamespace()
    ctx.invocation_metadata = lambda: meta_list
    return ctx


# ── 1. Token Extraction ──────────────────────────────────────────────────


class TestExtractRegistrationToken:
    """Tests for extract_registration_token_string."""

    def test_extracts_bearer_token(self):
        """Standard 'Bearer <token>' header is extracted correctly."""
        ctx = _make_grpc_context({"authorization": "Bearer abc123.payload.sig"})
        result = extract_registration_token_string(ctx)
        assert result == "abc123.payload.sig"

    def test_missing_auth_header_returns_empty(self):
        """No authorization header returns empty string.

        Kills mutant: return "" → return "XXXX"
        """
        ctx = _make_grpc_context({})
        result = extract_registration_token_string(ctx)
        assert result == ""

    def test_bearer_case_sensitive(self):
        """'bearer' (lowercase) does NOT match — must be 'Bearer '."""
        ctx = _make_grpc_context({"authorization": "bearer lowercase-token"})
        result = extract_registration_token_string(ctx)
        assert result == ""


# ── 2. Token Structure Validation ─────────────────────────────────────────


class TestBuildRegistrationVerifier:
    """Tests for build_registration_verifier structure validation."""

    @pytest.mark.asyncio
    async def test_rejects_too_few_parts(self):
        """Token with <3 dot-separated parts raises SecurityError.

        Kills mutant: .rsplit(".", 2) → .rsplit(".")
        """
        with pytest.raises(SecurityError):
            await build_registration_verifier(
                token_str="only-two.parts",
                project_id="test-project",
            )

    @pytest.mark.asyncio
    async def test_rejects_no_colon_in_kid(self):
        """KID without ':' separator raises SecurityError (not composite format)."""
        with pytest.raises(SecurityError):
            await build_registration_verifier(
                token_str="nokidcolon.payload.signature",
                project_id="test-project",
            )

    @pytest.mark.asyncio
    async def test_rejects_project_mismatch(self):
        """Token with mismatched project_id raises SecurityError."""
        with pytest.raises(SecurityError, match="mismatch"):
            await build_registration_verifier(
                token_str="wrong-project:v1.payload.signature",
                project_id="expected-project",
            )

    @pytest.mark.asyncio
    async def test_single_part_rejected(self):
        """Single token string without any dots raises SecurityError."""
        with pytest.raises(SecurityError):
            await build_registration_verifier(
                token_str="nodots",
                project_id="test-project",
            )

    @pytest.mark.asyncio
    async def test_hmac_without_project_secret_fails_closed(self, monkeypatch):
        """HMAC registration must not fall back to a global Router secret."""
        monkeypatch.setattr(
            "contextunity.core.discovery.get_project_key",
            lambda project_id: {},
        )

        with pytest.raises(SecurityError, match="project-scoped HMAC secret"):
            await build_registration_verifier(
                token_str="test-project:hmac-001.payload.signature",
                project_id="test-project",
            )
