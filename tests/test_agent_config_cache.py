"""Tests for agent config cache and permission intersection."""

import pytest

from contextrouter.core.agent_config_cache import (
    AgentConfigCache,
    AgentPermissionProfile,
    intersect_permissions,
)


class TestIntersectPermissions:
    """Test permission intersection logic."""

    def test_full_overlap(self):
        agent_max = ("brain:read", "graph:rag", "memory:read")
        request = ("brain:read", "graph:rag", "memory:read")
        result = intersect_permissions(agent_max, request)
        assert result == ("brain:read", "graph:rag", "memory:read")

    def test_partial_overlap(self):
        agent_max = ("brain:read", "graph:rag")
        request = ("brain:read", "brain:write", "graph:rag")
        result = intersect_permissions(agent_max, request)
        assert result == ("brain:read", "graph:rag")
        assert "brain:write" not in result

    def test_no_overlap(self):
        agent_max = ("graph:commerce",)
        request = ("graph:rag", "brain:read")
        result = intersect_permissions(agent_max, request)
        assert result == ()

    def test_empty_agent(self):
        result = intersect_permissions((), ("brain:read",))
        assert result == ()

    def test_empty_request(self):
        result = intersect_permissions(("brain:read",), ())
        assert result == ()

    def test_both_empty(self):
        result = intersect_permissions((), ())
        assert result == ()

    def test_sorted_output(self):
        agent_max = ("memory:read", "brain:read", "graph:rag")
        request = ("graph:rag", "brain:read", "memory:read")
        result = intersect_permissions(agent_max, request)
        assert result == tuple(sorted(result))


class TestAgentPermissionProfile:
    """Test AgentPermissionProfile dataclass."""

    def test_frozen(self):
        profile = AgentPermissionProfile(agent_id="test")
        with pytest.raises(AttributeError):
            profile.agent_id = "changed"

    def test_defaults(self):
        profile = AgentPermissionProfile(agent_id="test")
        assert profile.permissions == ()
        assert profile.allowed_tools == ()
        assert profile.denied_tools == ()
        assert profile.profile_name == ""


class TestAgentConfigCache:
    """Test AgentConfigCache behavior."""

    @pytest.mark.asyncio
    async def test_no_endpoint_uses_defaults(self):
        """Without admin_endpoint, cache returns default permissions."""
        cache = AgentConfigCache(admin_endpoint="")
        default_perms = ("brain:read", "graph:rag")
        profile = await cache.get_agent_permissions("test-agent", default_permissions=default_perms)
        assert profile.agent_id == "test-agent"
        # expand_permissions may add inherited perms
        assert "brain:read" in profile.permissions
        assert "graph:rag" in profile.permissions

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Second call should return cached result."""
        cache = AgentConfigCache(admin_endpoint="", ttl_seconds=60.0)
        default_perms = ("brain:read",)

        profile1 = await cache.get_agent_permissions("agent-a", default_permissions=default_perms)
        profile2 = await cache.get_agent_permissions("agent-a", default_permissions=default_perms)
        # Same object (frozen dataclass, same fetched_at)
        assert profile1.fetched_at == profile2.fetched_at

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Invalidation forces re-fetch."""
        cache = AgentConfigCache(admin_endpoint="", ttl_seconds=60.0)
        default_perms = ("brain:read",)

        profile1 = await cache.get_agent_permissions("agent-b", default_permissions=default_perms)
        cache.invalidate("agent-b")
        profile2 = await cache.get_agent_permissions("agent-b", default_permissions=default_perms)
        # Different fetch timestamps
        assert profile2.fetched_at >= profile1.fetched_at

    @pytest.mark.asyncio
    async def test_invalidate_all(self):
        """Clear all cache entries."""
        cache = AgentConfigCache(admin_endpoint="", ttl_seconds=60.0)
        await cache.get_agent_permissions("a", default_permissions=("brain:read",))
        await cache.get_agent_permissions("b", default_permissions=("brain:read",))
        assert len(cache._cache) == 2
        cache.invalidate_all()
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    async def test_different_agents_different_entries(self):
        """Each agent_id gets its own cache entry."""
        cache = AgentConfigCache(admin_endpoint="")
        await cache.get_agent_permissions("agent-1", default_permissions=("brain:read",))
        await cache.get_agent_permissions("agent-2", default_permissions=("graph:rag",))
        assert "agent-1" in cache._cache
        assert "agent-2" in cache._cache
