"""Persistence mixin — Redis-backed registration recovery."""

from __future__ import annotations

import json

from contextunity.core import get_contextunit_logger

from contextunity.router.core import get_core_config
from contextunity.router.service.payloads import GraphConfig, ToolConfig
from contextunity.router.service.tool_factory import create_tool_from_config

logger = get_contextunit_logger(__name__)

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

    async def _check_manifest_hash(self, project_id: str, current_hash: str) -> bool:
        """Check if the given manifest hash matches the one stored in Redis."""
        if not current_hash:
            return False

        r = await self._get_redis()
        if not r:
            return False
        try:
            stored_hash = await r.get(f"{_REDIS_PREFIX}:{project_id}:hash")
            return stored_hash == current_hash
        except Exception as e:
            logger.warning("Failed to check manifest hash in Redis: %s", e)
            return False
        finally:
            await r.aclose()

    async def _save_manifest_hash(self, project_id: str, new_hash: str) -> None:
        """Save the new manifest hash to Redis."""
        if not new_hash:
            return

        r = await self._get_redis()
        if not r:
            return
        try:
            await r.set(f"{_REDIS_PREFIX}:{project_id}:hash", new_hash)
        except Exception as e:
            logger.warning("Failed to save manifest hash to Redis: %s", e)
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
                    project_id = payload.get("project_id")
                    if not project_id:
                        logger.warning("Persisted payload missing 'project_id', skipping.")
                        continue
                    tenant_id = payload.get("tenant_id", project_id)

                    # Deregister previous (idempotent)
                    if project_id in self._project_tools:
                        self._deregister_project(project_id)

                    # Re-create tools
                    registered_tools: list[str] = []
                    for tool_dict in payload.get("tools", []):
                        tool_def = ToolConfig(**tool_dict)
                        tools = create_tool_from_config(
                            name=tool_def.name,
                            tool_type=tool_def.type,
                            description=tool_def.description,
                            config=tool_def.config,
                        )
                        from contextunity.router.modules.tools import register_tool

                        for tool_instance in tools:
                            register_tool(tool_instance, tenant=tenant_id)
                            registered_tools.append(tool_instance.name)

                    self._project_tools[project_id] = registered_tools

                    # Re-register graph
                    graph_name = None
                    graph_dict = payload.get("graph")
                    if graph_dict and "id" in graph_dict:
                        graph_config = GraphConfig(
                            name=graph_dict["id"],
                            template=graph_dict.get("template"),
                            nodes=graph_dict.get("nodes", []),
                            edges=graph_dict.get("edges", []),
                            config=graph_dict.get("config", {}),
                        )
                        graph_name = self._register_graph(project_id, graph_config)

                        if project_id not in self._project_configs:
                            self._project_configs[project_id] = {}
                        self._project_configs[project_id]["nodes"] = graph_dict.get("nodes", [])
                        self._project_configs[project_id]["policy"] = payload.get("policy", {})
                        self._project_configs[project_id]["tools"] = payload.get("tools", [])

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
