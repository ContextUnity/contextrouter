"""RED phase: Tests for make_platform_node() dispatch through PlatformToolRegistry.

Verifies that platform.py correctly looks up tools from the registry,
validates config, checks scopes, and dispatches execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextunity.core.exceptions import SecurityError
from pydantic import BaseModel

from contextunity.router.cortex.compiler.node_executors.platform import (
    make_platform_node,
)


class MockToolConfig(BaseModel):
    scope_path: str = "default.scope"
    ttl_seconds: int = 300


def _make_state(**kwargs):
    state = {
        "tenant_id": "test_tenant",
        "messages": [],
        "__token__": MagicMock(permissions=["brain:write", "brain:read"]),
    }
    state.update(kwargs)
    return state


@pytest.mark.asyncio
async def test_platform_node_dispatches_through_registry():
    """Platform node should lookup tool in registry and call its executor."""
    node_spec = {
        "name": "write_blackboard",
        "type": "tool",
        "tool_binding": "brain_blackboard_write",
        "config": {"scope_path": "test.session", "ttl_seconds": 60},
    }

    mock_executor = AsyncMock(return_value={"id": "uuid-123", "status": "written"})

    mock_registry = MagicMock()
    mock_registration = MagicMock()
    mock_registration.executor = mock_executor
    mock_registration.config_schema = MockToolConfig
    mock_registration.required_scopes = ["brain:write"]
    mock_registry.get.return_value = mock_registration
    mock_registry.validate_config.return_value = MockToolConfig(
        scope_path="test.session", ttl_seconds=60
    )
    mock_registry.check_scopes.return_value = None

    with patch(
        "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
        return_value=mock_registry,
    ):
        executor = make_platform_node(node_spec, {})
        result = await executor(_make_state(), {})

    assert "final_output" in result
    assert result["final_output"]["id"] == "uuid-123"
    mock_executor.assert_called_once()
    mock_registry.check_scopes.assert_called_once()


@pytest.mark.asyncio
async def test_platform_node_missing_scope_raises():
    """Platform node rejects execution when token lacks required scopes."""
    node_spec = {
        "name": "write_blackboard",
        "type": "tool",
        "tool_binding": "brain_blackboard_write",
        "config": {},
    }

    mock_registry = MagicMock()
    mock_registry.get.return_value = MagicMock(required_scopes=["brain:write"])
    mock_registry.check_scopes.side_effect = SecurityError(message="Missing scope brain:write")

    with patch(
        "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
        return_value=mock_registry,
    ):
        executor = make_platform_node(node_spec, {})

        with pytest.raises(SecurityError, match="brain:write"):
            await executor(_make_state(), {})


@pytest.mark.asyncio
async def test_platform_node_routes_output_to_config_key():
    """Platform node should respect config state_output_key."""
    node_spec = {
        "name": "search_vectors",
        "type": "tool",
        "tool_binding": "brain_search",
        "config": {"state_output_key": "search_results", "top_k": 5},
    }

    mock_executor = AsyncMock(return_value={"matches": [{"id": "doc-1"}]})

    mock_registry = MagicMock()
    mock_registration = MagicMock()
    mock_registration.executor = mock_executor
    mock_registration.config_schema = MockToolConfig
    mock_registration.required_scopes = ["brain:read"]
    mock_registry.get.return_value = mock_registration
    mock_registry.validate_config.return_value = MockToolConfig()
    mock_registry.check_scopes.return_value = None

    with patch(
        "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
        return_value=mock_registry,
    ):
        executor = make_platform_node(node_spec, {})
        result = await executor(_make_state(), {})

    # Output should route to custom key
    assert "search_results" in result
    assert result["search_results"]["matches"][0]["id"] == "doc-1"
