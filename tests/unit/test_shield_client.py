"""Behavioral tests for service.shield_client.

Tests cover:
  - shield_verify_secret: constant-time comparison, None stored, mismatch
"""

from __future__ import annotations

from contextunity.router.service.shield_client import (
    shield_verify_secret,
)


class TestShieldVerifySecret:
    def test_returns_false_when_no_stored_secret(self, monkeypatch):
        """No stored secret → False (not an exception)."""
        monkeypatch.setattr(
            "contextunity.router.service.shield_client.shield_get_secret",
            lambda path, **kw: None,
        )
        assert shield_verify_secret("path/key", "candidate") is False

    def test_matching_secret_returns_true(self, monkeypatch):
        """Stored secret matches candidate → True."""
        monkeypatch.setattr(
            "contextunity.router.service.shield_client.shield_get_secret",
            lambda path, **kw: "correct-secret",
        )
        assert shield_verify_secret("path/key", "correct-secret") is True

    def test_mismatched_secret_returns_false(self, monkeypatch):
        """Stored secret does not match candidate → False."""
        monkeypatch.setattr(
            "contextunity.router.service.shield_client.shield_get_secret",
            lambda path, **kw: "correct-secret",
        )
        assert shield_verify_secret("path/key", "wrong-secret") is False
