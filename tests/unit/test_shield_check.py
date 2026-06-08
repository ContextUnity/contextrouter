"""Tests for Shield inline content check.

Covers:
  - URL resolution: valid, blank, cache, config error fallback
  - check_user_input: passthrough (no shield), fail-closed (shield unreachable),
    mode field verification
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import contextunity.router.service.shield_check as shield_check


@pytest.fixture(autouse=True)
def _reset_shield_state():
    """Reset module-level cache before each test."""
    shield_check._shield_url = None
    shield_check._shield_url_resolved = False
    shield_check._shield_channel = None
    yield
    shield_check._shield_url = None
    shield_check._shield_url_resolved = False
    shield_check._shield_channel = None


# ── URL Resolution ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("shield_url", "expected"),
    [
        ("shield.prod:50054", "shield.prod:50054"),
        ("localhost:50054", "localhost:50054"),
        ("  ", ""),
        (None, ""),
    ],
)
def test_get_shield_url_resolves_correctly(monkeypatch, shield_url, expected):
    monkeypatch.setattr(
        "contextunity.router.core.get_core_config",
        lambda: SimpleNamespace(shield_url=shield_url),
    )
    assert shield_check._get_shield_url() == expected


def test_get_shield_url_config_error_returns_empty(monkeypatch):
    """If config import/call fails, return empty (disabled)."""
    monkeypatch.setattr(
        "contextunity.router.core.get_core_config",
        lambda: (_ for _ in ()).throw(RuntimeError("config broken")),
    )
    assert shield_check._get_shield_url() == ""


# ── check_user_input ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_passthrough_when_no_shield(monkeypatch):
    """No shield URL → passthrough, not blocked, mode='passthrough'."""
    monkeypatch.setattr(shield_check, "_get_shield_url", lambda: "")

    result = await shield_check.check_user_input("hello")

    assert result.blocked is False
    assert result.mode == "passthrough"
    assert result.reason == ""


@pytest.mark.asyncio
async def test_fail_closed_on_remote_shield_error(monkeypatch):
    """Shield configured but unreachable → BLOCKED (fail-closed)."""
    monkeypatch.setattr(shield_check, "_get_shield_url", lambda: "shield.prod:50054")
    monkeypatch.setattr(
        shield_check,
        "_get_shield_channel",
        lambda url: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )

    result = await shield_check.check_user_input("hello", request_id="req-1", tenant="t-a")

    assert result.blocked is True
    assert result.reason == "Shield unavailable"
    assert result.mode == "shield"


@pytest.mark.asyncio
async def test_fail_closed_on_localhost_shield_error(monkeypatch):
    """Even localhost Shield failure blocks — no special-casing."""
    monkeypatch.setattr(shield_check, "_get_shield_url", lambda: "localhost:50054")
    monkeypatch.setattr(
        shield_check,
        "_get_shield_channel",
        lambda url: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = await shield_check.check_user_input("hello", request_id="r-1", tenant="t-b")

    assert result.blocked is True
    assert result.mode == "shield"


# ── Mutation Killers ──────────────────────────────────────────────────────
