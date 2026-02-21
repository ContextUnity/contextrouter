"""Persistence mixin — Redis-backed registration recovery."""

from __future__ import annotations

import json

from contextcore import get_context_unit_logger

from contextrouter.core import get_core_config
from contextrouter.service.payloads import RegisterToolsPayload
from contextrouter.service.tool_factory import create_tool_from_config

logger = get_context_unit_logger(__name__)

# Redis key prefix for persisted registrations
_REDIS_PREFIX = "router:registrations"


class PersistenceMixin:
    """Mixin providing Redis-backed registration persistence and recovery."""

    async def _get_redis(self):
        """Get async Redis client from Router config."""
        try:
            import redis.asyncio as aioredis

            config = get_core_config()
            url = config.redis.url
            return aioredis.from_url(url, decode_responses=True)
        except Exception as e:
            logger.warning("Redis not available for persistence: %s", e)
            return None

    async def _persist_registration(self, project_id: str, payload: dict):
        """Save registration payload to Redis for restart recovery."""
        r = await self._get_redis()
        if not r:
            return
        try:
            key = f"{_REDIS_PREFIX}:{project_id}"
            await r.set(key, json.dumps(payload, default=str))
            logger.debug("Persisted registration for '%s' to Redis", project_id)
        except Exception as e:
            logger.warning("Failed to persist registration to Redis: %s", e)
        finally:
            await r.aclose()

    async def _remove_persisted_registration(self, project_id: str):
        """Remove persisted registration from Redis."""
        r = await self._get_redis()
        if not r:
            return
        try:
            await r.delete(f"{_REDIS_PREFIX}:{project_id}")
            logger.debug("Removed persisted registration for '%s'", project_id)
        except Exception as e:
            logger.warning("Failed to remove registration from Redis: %s", e)
        finally:
            await r.aclose()

    async def restore_registrations(self):
        """Restore all project registrations from Redis.

        Called once during server startup to recover state after restart.
        """
        r = await self._get_redis()
        if not r:
            logger.warning("Redis not available — no registrations to restore")
            return

        try:
            keys = []
            async for key in r.scan_iter(f"{_REDIS_PREFIX}:*"):
                keys.append(key)

            if not keys:
                logger.info("No persisted registrations found in Redis")
                return

            logger.info("Restoring %d project registration(s) from Redis...", len(keys))

            for key in keys:
                project_id = key.split(":", 2)[-1]
                try:
                    raw = await r.get(key)
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    params = RegisterToolsPayload(**payload)

                    # Deregister previous (idempotent)
                    if params.project_id in self._project_tools:
                        self._deregister_project(params.project_id)

                    # Re-create tools
                    registered_tools: list[str] = []
                    for tool_def in params.tools:
                        tools = create_tool_from_config(
                            name=tool_def.name,
                            tool_type=tool_def.type,
                            description=tool_def.description,
                            config=tool_def.config,
                        )
                        from contextrouter.modules.tools import register_tool

                        for tool_instance in tools:
                            register_tool(tool_instance)
                            registered_tools.append(tool_instance.name)

                    self._project_tools[params.project_id] = registered_tools

                    # Re-register graph
                    graph_name = None
                    if params.graph:
                        graph_name = self._register_graph(params.project_id, params.graph)

                    logger.info(
                        "Restored project '%s': %d tools, graph=%s",
                        project_id,
                        len(registered_tools),
                        graph_name or "none",
                    )
                except Exception as e:
                    logger.warning(
                        "Skipped restoring project '%s' (secrets may not be available yet): %s",
                        project_id,
                        e,
                    )
        except Exception as e:
            logger.error("Failed to scan Redis for registrations: %s", e)
        finally:
            await r.aclose()


__all__ = ["PersistenceMixin"]
