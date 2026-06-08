"""RED phase: Tests for Brain platform tool executors.

Tests define the contracts for brain_search, brain_memory_read/write,
brain_blackboard_read/write, brain_kg_query, brain_upsert executors
before any implementation exists.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from contextunity.core.tokens import ContextToken


def _make_state(**kwargs):
    state = {
        "tenant_id": "test_tenant",
        "messages": [],
        "__token__": ContextToken(
            token_id="brain-tool-test",
            user_id="test-user",
            allowed_tenants=("test_tenant",),
            permissions=("brain:read", "brain:write", "memory:read", "memory:write"),
        ),
    }
    state.update(kwargs)
    return state


# ── Registration Tests ──────────────────────────────────────────────


# ── Executor Tests ──────────────────────────────────────────────────


class TestBrainSearchExecutor:
    """Test brain_search tool executor."""

    @pytest.mark.asyncio
    async def test_calls_brain_client_search(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )
        from contextunity.router.cortex.compiler.platform_tools.brain import (
            register_brain_tools,
        )

        registry = PlatformToolRegistry()
        register_brain_tools(registry)
        registration = registry.get("brain_search")

        mock_client = AsyncMock()
        mock_client.search.return_value = [{"id": "doc-1", "score": 0.95, "content": "test"}]

        with patch(
            "contextunity.router.cortex.compiler.platform_tools.brain.executors._get_brain_client",
            return_value=mock_client,
        ):
            from contextunity.router.cortex.compiler.platform_tools.brain import (
                BrainSearchConfig,
            )

            config = BrainSearchConfig(top_k=5, collection="test_collection")
            result = await registration.executor(
                _make_state(messages=[{"role": "user", "content": "find docs"}]),
                config,
            )

        assert "results" in result
        assert len(result["results"]) == 1
        mock_client.search.assert_called_once()


class TestBrainBlackboardWriteExecutor:
    """Test brain_blackboard_write tool executor."""

    @pytest.mark.asyncio
    async def test_writes_to_blackboard(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )
        from contextunity.router.cortex.compiler.platform_tools.brain import (
            register_brain_tools,
        )

        registry = PlatformToolRegistry()
        register_brain_tools(registry)
        registration = registry.get("brain_blackboard_write")

        mock_client = AsyncMock()
        mock_client.write_blackboard.return_value = {
            "id": "uuid-123",
            "scope_path": "test.session",
        }

        with patch(
            "contextunity.router.cortex.compiler.platform_tools.brain.executors._get_brain_client",
            return_value=mock_client,
        ):
            from contextunity.router.cortex.compiler.platform_tools.brain import (
                BrainBlackboardWriteConfig,
            )

            config = BrainBlackboardWriteConfig(scope_path="test.session", ttl_seconds=300)
            result = await registration.executor(
                _make_state(classification={"key": "value"}),
                config,
            )

        assert "id" in result
        mock_client.write_blackboard.assert_called_once()


class TestBrainBlackboardReadExecutor:
    """Test brain_blackboard_read tool executor."""

    @pytest.mark.asyncio
    async def test_reads_from_blackboard(self):
        from contextunity.router.cortex.compiler.platform_registry import (
            PlatformToolRegistry,
        )
        from contextunity.router.cortex.compiler.platform_tools.brain import (
            register_brain_tools,
        )

        registry = PlatformToolRegistry()
        register_brain_tools(registry)
        registration = registry.get("brain_blackboard_read")

        mock_client = AsyncMock()
        mock_client.read_blackboard.return_value = {
            "records": [{"id": "uuid-123", "content": {"data": 42}}]
        }

        with patch(
            "contextunity.router.cortex.compiler.platform_tools.brain.executors._get_brain_client",
            return_value=mock_client,
        ):
            from contextunity.router.cortex.compiler.platform_tools.brain import (
                BrainBlackboardReadConfig,
            )

            config = BrainBlackboardReadConfig(
                ids=["uuid-123"],
            )
            result = await registration.executor(_make_state(), config)

        assert "records" in result
        mock_client.read_blackboard.assert_called_once()


# ── Config Validation Tests ─────────────────────────────────────────


class TestBrainToolConfigs:
    """Test config schemas for brain tools."""

    def test_search_config_defaults(self):
        from contextunity.router.cortex.compiler.platform_tools.brain import (
            BrainSearchConfig,
        )

        config = BrainSearchConfig()
        assert config.top_k == 10
        assert config.collection == "default"

    def test_blackboard_write_config_requires_scope(self):
        from contextunity.router.cortex.compiler.platform_tools.brain import (
            BrainBlackboardWriteConfig,
        )

        config = BrainBlackboardWriteConfig(scope_path="test.path")
        assert config.scope_path == "test.path"
        assert config.ttl_seconds is None

    def test_blackboard_read_config_requires_ids(self):
        from contextunity.router.cortex.compiler.platform_tools.brain import (
            BrainBlackboardReadConfig,
        )

        with pytest.raises(Exception):
            BrainBlackboardReadConfig()  # ids is required
