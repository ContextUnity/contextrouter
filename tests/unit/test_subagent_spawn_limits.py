"""Spawn-limit tests for subagent tools.

Pins two contracts:
1. A caller already nested MAX_SUBAGENT_DEPTH deep cannot spawn further.
2. A (tenant, session) pair cannot exceed MAX_SPAWNS_PER_SESSION spawns.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from contextunity.core.tokens import ContextToken

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.modules.tools import subagent_tools


@pytest.fixture(autouse=True)
def _reset_spawn_counts():
    subagent_tools._spawn_counts.clear()
    yield
    subagent_tools._spawn_counts.clear()


@pytest.fixture
def _mock_spawner():
    spawner = AsyncMock()
    spawner.spawn_subagent.return_value = "subagent-1"
    with patch.object(subagent_tools, "_spawner", spawner):
        yield spawner


def _nested_token(depth: int) -> ContextToken:
    provenance = tuple(f">subagent:worker{i}" for i in range(depth))
    return ContextToken(
        token_id="t",
        permissions=("brain:read",),
        allowed_tenants=("tenant_a",),
        provenance=provenance,
    )


@pytest.mark.asyncio
async def test_depth_limit_refuses_deeply_nested_caller(_mock_spawner):
    token_ref = set_current_access_token(_nested_token(subagent_tools.MAX_SUBAGENT_DEPTH))
    try:
        result = await subagent_tools.spawn_subagent.ainvoke(
            {"task": {"description": "x"}, "tenant_id": "tenant_a", "session_id": "s1"}
        )
    finally:
        reset_current_access_token(token_ref)
    assert result["success"] is False
    assert result["error"] == "max_depth_exceeded"
    _mock_spawner.spawn_subagent.assert_not_called()


@pytest.mark.asyncio
async def test_depth_below_limit_allows_spawn(_mock_spawner):
    token_ref = set_current_access_token(_nested_token(subagent_tools.MAX_SUBAGENT_DEPTH - 1))
    try:
        result = await subagent_tools.spawn_subagent.ainvoke(
            {"task": {"description": "x"}, "tenant_id": "tenant_a", "session_id": "s1"}
        )
    finally:
        reset_current_access_token(token_ref)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_session_spawn_budget_exhausts(_mock_spawner, monkeypatch):
    monkeypatch.setattr(subagent_tools, "MAX_SPAWNS_PER_SESSION", 2)
    args = {"task": {"description": "x"}, "tenant_id": "tenant_a", "session_id": "s-budget"}

    token_ref = set_current_access_token(_nested_token(0))
    try:
        first = await subagent_tools.spawn_subagent.ainvoke(args)
        second = await subagent_tools.spawn_subagent.ainvoke(args)
        third = await subagent_tools.spawn_subagent.ainvoke(args)
    finally:
        reset_current_access_token(token_ref)

    assert first["success"] is True
    assert second["success"] is True
    assert third["success"] is False
    assert third["error"] == "spawn_budget_exhausted"


@pytest.mark.asyncio
async def test_budget_is_per_session(_mock_spawner, monkeypatch):
    monkeypatch.setattr(subagent_tools, "MAX_SPAWNS_PER_SESSION", 1)
    base = {"task": {"description": "x"}, "tenant_id": "tenant_a"}

    token_ref = set_current_access_token(_nested_token(0))
    try:
        first = await subagent_tools.spawn_subagent.ainvoke({**base, "session_id": "s1"})
        other_session = await subagent_tools.spawn_subagent.ainvoke({**base, "session_id": "s2"})
    finally:
        reset_current_access_token(token_ref)

    assert first["success"] is True
    assert other_session["success"] is True


@pytest.mark.asyncio
async def test_spawn_refuses_cross_tenant_request(_mock_spawner):
    token_ref = set_current_access_token(_nested_token(0))
    try:
        result = await subagent_tools.spawn_subagent.ainvoke(
            {"task": {"description": "x"}, "tenant_id": "tenant_b", "session_id": "s1"}
        )
    finally:
        reset_current_access_token(token_ref)
    assert result["success"] is False
    assert result["error"] == "tenant_access_denied"
    _mock_spawner.spawn_subagent.assert_not_called()
