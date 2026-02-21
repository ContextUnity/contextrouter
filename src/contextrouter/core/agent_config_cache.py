"""Agent configuration cache for permission-aware token minting.

Fetches agent permission profiles from ContextView (AdminService)
and caches them in-memory with TTL. Used by chat runners to mint
tokens with agent-specific (not just default) permissions.

Cache key: ``agent_id`` → ``AgentPermissionProfile``
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from contextcore.permissions import PROJECT_PROFILES, expand_permissions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentPermissionProfile:
    """Cached permission profile for a registered agent.

    Attributes:
        agent_id: Unique agent identifier.
        permissions: Expanded permission tuple for this agent.
        allowed_tools: Tool whitelist (empty = all allowed).
        denied_tools: Tool blacklist.
        profile_name: Named profile (e.g. "rag_full", "commerce").
        fetched_at: Unix timestamp when fetched from ContextView.
    """

    agent_id: str
    permissions: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    profile_name: str = ""
    fetched_at: float = 0.0


@dataclass
class AgentConfigCache:
    """In-memory cache for agent permission profiles.

    Lazily fetches from ContextView AdminService and caches with TTL.
    Falls back to default permissions on error.
    """

    admin_endpoint: str = ""
    ttl_seconds: float = 300.0  # 5 minutes
    _cache: dict[str, AgentPermissionProfile] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_agent_permissions(
        self,
        agent_id: str,
        default_permissions: tuple[str, ...] = (),
    ) -> AgentPermissionProfile:
        """Get cached agent permissions, fetching from ContextView if stale.

        Args:
            agent_id: Agent identifier to look up.
            default_permissions: Fallback permissions if agent not found.

        Returns:
            AgentPermissionProfile with expanded permissions.
        """
        now = time.monotonic()

        # Check cache (no lock needed for reads)
        cached = self._cache.get(agent_id)
        if cached and (now - cached.fetched_at) < self.ttl_seconds:
            return cached

        # Fetch under lock to avoid thundering herd
        async with self._lock:
            # Double-check after acquiring lock
            cached = self._cache.get(agent_id)
            if cached and (now - cached.fetched_at) < self.ttl_seconds:
                return cached

            profile = await self._fetch_agent_profile(agent_id, default_permissions)
            self._cache[agent_id] = profile
            return profile

    async def _fetch_agent_profile(
        self,
        agent_id: str,
        default_permissions: tuple[str, ...],
    ) -> AgentPermissionProfile:
        """Fetch agent config from ContextView AdminService.

        Falls back to default_permissions on any error.
        """
        now = time.monotonic()

        if not self.admin_endpoint:
            logger.debug(
                "No admin_endpoint configured, using default permissions for agent=%s",
                agent_id,
            )
            return AgentPermissionProfile(
                agent_id=agent_id,
                permissions=expand_permissions(default_permissions),
                fetched_at=now,
            )

        try:
            from contextunity.api.admin_client import AdminClient

            client = AdminClient(endpoint=self.admin_endpoint, mode="grpc")
            result = await client.get_agent_config(agent_id=agent_id)

            if not result or "error" in result:
                logger.warning(
                    "Agent %s not found in ContextView, using defaults",
                    agent_id,
                )
                return AgentPermissionProfile(
                    agent_id=agent_id,
                    permissions=expand_permissions(default_permissions),
                    fetched_at=now,
                )

            # Agent found — extract permissions
            agent_data = result.get("agent", result)
            raw_perms = tuple(agent_data.get("permissions", []))
            profile_name = agent_data.get("config", {}).get("profile", "")

            # If agent has a named profile, use it as base
            if profile_name and profile_name in PROJECT_PROFILES:
                raw_perms = raw_perms or PROJECT_PROFILES[profile_name]

            permissions = expand_permissions(raw_perms or default_permissions)
            allowed_tools = tuple(agent_data.get("allowed_tools", []))
            denied_tools = tuple(agent_data.get("denied_tools", []))

            logger.info(
                "Cached agent permissions: agent=%s profile=%s perms=%d",
                agent_id,
                profile_name or "(custom)",
                len(permissions),
            )

            return AgentPermissionProfile(
                agent_id=agent_id,
                permissions=permissions,
                allowed_tools=allowed_tools,
                denied_tools=denied_tools,
                profile_name=profile_name,
                fetched_at=now,
            )

        except ImportError:
            logger.debug(
                "AdminClient not available, using defaults for agent=%s",
                agent_id,
            )
            return AgentPermissionProfile(
                agent_id=agent_id,
                permissions=expand_permissions(default_permissions),
                fetched_at=now,
            )
        except Exception:
            logger.warning(
                "Failed to fetch agent config for %s, using defaults",
                agent_id,
                exc_info=True,
            )
            return AgentPermissionProfile(
                agent_id=agent_id,
                permissions=expand_permissions(default_permissions),
                fetched_at=now,
            )

    def invalidate(self, agent_id: str) -> None:
        """Remove an agent from cache, forcing re-fetch on next access."""
        self._cache.pop(agent_id, None)

    def invalidate_all(self) -> None:
        """Clear entire cache."""
        self._cache.clear()


def intersect_permissions(
    agent_max: tuple[str, ...],
    request_permissions: tuple[str, ...],
) -> tuple[str, ...]:
    """Compute effective permissions = intersection of agent max and request.

    The token gets only permissions that are both:
    1. Allowed by the agent's profile (ceiling)
    2. Requested by the current operation

    Args:
        agent_max: Maximum permissions the agent is allowed.
        request_permissions: Permissions requested for this operation.

    Returns:
        Sorted tuple of permissions in both sets.

    Example::

        agent_max = ("brain:read", "memory:read", "graph:rag")
        request = ("brain:read", "brain:write", "graph:rag")
        intersect_permissions(agent_max, request)
        # → ("brain:read", "graph:rag")  # brain:write excluded
    """
    return tuple(sorted(set(agent_max) & set(request_permissions)))


# Module-level singleton
_cache: AgentConfigCache | None = None


def get_agent_config_cache() -> AgentConfigCache:
    """Get or create the module-level agent config cache singleton."""
    global _cache
    if _cache is None:
        _cache = AgentConfigCache()
    return _cache


__all__ = [
    "AgentConfigCache",
    "AgentPermissionProfile",
    "get_agent_config_cache",
    "intersect_permissions",
]
