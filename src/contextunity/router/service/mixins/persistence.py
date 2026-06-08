"""Persistence mixin — Redis-backed registration recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, ContextUnityError
from contextunity.core.manifest.tenants import resolve_bundle_allowed_tenants
from contextunity.core.types import JsonDict, is_json_dict

from contextunity.router.service.mixins.execution.types import (
    ProjectConfigMap,
    ProjectGraphMap,
    ProjectToolMap,
    RouterCallbackMap,
)
from contextunity.router.service.payloads import GraphEntry
from contextunity.router.service.registration_projection import (
    registered_project_config_from_persisted,
)
from contextunity.router.service.registration_redis import (
    RegistrationRedisStore,
    open_registration_redis_store,
)
from contextunity.router.service.registration_tools import create_tools_from_bundle

logger = get_contextunit_logger(__name__)
_REGISTRATION_HASH_FIELD = "__registration_hash"


def _validate_graph_key_field_consistency(
    graph_key: str,
    field_name: str,
    field_value: str | None,
) -> None:
    if field_value is not None and field_value != graph_key:
        raise ConfigurationError(
            message=(
                f"Graph entry {field_name} '{field_value}' does not match its map key "
                f"'{graph_key}'. The map key is the canonical graph id."
            )
        )


class PersistenceMixin:
    """Mixin providing Redis-backed registration persistence and recovery."""

    _project_graphs: ProjectGraphMap = {}
    _project_configs: ProjectConfigMap = {}
    _project_tools: ProjectToolMap = {}
    _project_router_callbacks: RouterCallbackMap = {}

    if TYPE_CHECKING:

        def _deregister_project(self, _project_id: str) -> list[str]: ...
        def _register_graph(
            self,
            _project_id: str,
            _graph_config: GraphEntry,
            *,
            available_tools: set[str] | None = None,
        ) -> str:
            del available_tools
            raise NotImplementedError

    async def _open_registration_store(self) -> RegistrationRedisStore | None:
        """Open typed registration Redis store (override in tests)."""
        return await open_registration_redis_store()

    async def _persist_registration(self, project_id: str, payload: dict[str, object]) -> None:
        """Save registration payload to Redis for restart recovery."""
        store = await self._open_registration_store()
        if not store:
            return
        try:
            await store.persist(project_id, payload)
            logger.debug("Persisted registration for '%s' to Redis", project_id)
        except Exception as e:  # graceful-degrade: Redis optional, continue without
            logger.warning("Failed to persist registration to Redis: %s", e)
        finally:
            await store.close()

    async def _remove_persisted_registration(self, project_id: str):
        """Remove persisted registration (and associated hash) from Redis."""
        store = await self._open_registration_store()
        if not store:
            return
        try:
            deleted = await store.remove(project_id)
            logger.debug(
                "Removed persisted registration for '%s' (deleted=%s)", project_id, deleted
            )
        except Exception as e:  # graceful-degrade: Redis optional, continue without
            logger.warning("Failed to remove registration from Redis: %s", e)
        finally:
            await store.close()

    @staticmethod
    def _to_restore_error(project_id: str, exc: Exception) -> ContextUnityError:
        """Normalize restore failures to ContextUnityError for stable logging."""
        if isinstance(exc, ContextUnityError):
            return exc
        return ConfigurationError(
            message=f"Invalid persisted registration payload for project '{project_id}'",
            project_id=project_id,
            cause=str(exc),
        )

    @staticmethod
    def _rollback_registered_tools(project_id: str, tool_names: set[str]) -> None:
        """Remove project tools registered before a restore failure."""
        if not tool_names:
            return
        from contextunity.router.modules.tools import deregister_tool

        for tool_name in tool_names:
            _ = deregister_tool(tool_name, project_id=project_id)

    async def _check_manifest_hash(self, project_id: str, current_hash: str) -> bool:
        """Check the hash embedded atomically in an existing persisted payload."""
        if not current_hash:
            return False

        store = await self._open_registration_store()
        if not store:
            return False
        try:
            payload = await store.load_payload(store.project_key(project_id))
            if payload is None:
                return False
            embedded_hash = payload.get(_REGISTRATION_HASH_FIELD)
            stored_hash = embedded_hash if isinstance(embedded_hash, str) else None
            if stored_hash is None:
                stored_hash = await store.read_hash(project_id)
            return stored_hash == current_hash
        except Exception as e:  # graceful-degrade: Redis optional, continue without
            logger.warning("Failed to check manifest hash in Redis: %s", e)
            return False
        finally:
            await store.close()

    async def _save_manifest_hash(self, project_id: str, new_hash: str) -> None:
        """Save the new manifest hash to Redis."""
        if not new_hash:
            return

        store = await self._open_registration_store()
        if not store:
            return
        try:
            await store.write_hash(project_id, new_hash)
        except Exception as e:  # graceful-degrade: Redis optional, continue without
            logger.warning("Failed to save manifest hash to Redis: %s", e)
        finally:
            await store.close()

    async def _persist_stream_secret(self, project_id: str, stream_secret: str) -> None:
        """Persist BiDi stream auth secret under ``router:registrations:{id}:stream``."""
        store = await self._open_registration_store()
        if not store:
            return
        try:
            await store.write_stream_secret(project_id, stream_secret)
        except Exception as e:  # graceful-degrade: Redis optional, continue without
            logger.warning("Failed to persist stream secret in Redis: %s", e)
        finally:
            await store.close()

    async def restore_registrations(self):
        """Restore all project registrations from Redis.

        Called once during server startup to recover state after restart.
        """
        store = await self._open_registration_store()
        if not store:
            logger.warning("Redis not available — no registrations to restore")
            return

        try:
            keys = await store.list_registration_keys()
            if not keys:
                logger.debug("No persisted registrations found in Redis")
                return

            logger.info("Restoring %d project registration(s) from Redis...", len(keys))

            for key in keys:
                if key.endswith((":hash", ":stream")):
                    continue
                project_id = key.split(":", 2)[-1]
                registered_tools: set[str] = set()
                try:
                    payload = await store.load_payload(key)
                    if payload is None:
                        logger.warning("Persisted payload is not a JSON object, skipping.")
                        continue
                    project_id_raw = payload.get("project_id")
                    if not isinstance(project_id_raw, str) or not project_id_raw:
                        logger.warning("Persisted payload missing 'project_id', skipping.")
                        continue
                    project_id = project_id_raw
                    allowed_tenants = resolve_bundle_allowed_tenants(dict(payload))
                    tool_instances = create_tools_from_bundle(dict(payload), project_id=project_id)

                    if project_id in self._project_tools:
                        _ = self._deregister_project(project_id)

                    from contextunity.router.modules.tools import register_tool

                    for tool_instance in tool_instances:
                        register_tool(
                            tool_instance,
                            allowed_tenants=tuple(allowed_tenants),
                            project_id=project_id,
                        )
                        registered_tools.add(tool_instance.name)

                    graph_name: str | None = None
                    graph_map_raw = payload.get("graph")
                    graph_map: JsonDict = graph_map_raw if is_json_dict(graph_map_raw) else {}
                    if graph_map:
                        registered_graph_map: dict[str, str] = {}
                        callbacks_map: dict[str, list[str]] = {}

                        for graph_key, raw_entry in graph_map.items():
                            entry = GraphEntry.model_validate(raw_entry)
                            _validate_graph_key_field_consistency(graph_key, "id", entry.id)
                            _validate_graph_key_field_consistency(graph_key, "name", entry.name)
                            if entry.id is None:
                                entry = entry.model_copy(update={"id": graph_key})
                            entry = entry.model_copy(update={"name": graph_key})
                            g_name = self._register_graph(
                                project_id,
                                entry,
                                available_tools=set(registered_tools),
                            )
                            registered_graph_map[graph_key] = g_name
                            if entry.router_callbacks:
                                callbacks_map[graph_key] = list(entry.router_callbacks)

                        default_graph_id_raw = payload.get("default_graph")
                        default_graph_id = (
                            default_graph_id_raw if isinstance(default_graph_id_raw, str) else None
                        )
                        if default_graph_id and default_graph_id in registered_graph_map:
                            registered_graph_map["default"] = registered_graph_map[default_graph_id]
                        elif len(registered_graph_map) == 1:
                            registered_graph_map["default"] = next(
                                iter(registered_graph_map.values())
                            )
                        elif len(registered_graph_map) > 1:
                            raise ConfigurationError(
                                message=(
                                    "Persisted multi-graph registration missing valid default_graph"
                                )
                            )

                        self._project_graphs[project_id] = registered_graph_map
                        self._project_router_callbacks[project_id] = callbacks_map
                        graph_name = registered_graph_map.get("default")

                        self._project_configs[project_id] = (
                            registered_project_config_from_persisted(dict(payload), graph_map)
                        )

                    self._project_tools[project_id] = sorted(registered_tools)

                    logger.debug(
                        "Restored project '%s': %d tools, graph=%s",
                        project_id,
                        len(self._project_tools[project_id]),
                        graph_name or "none",
                    )
                except Exception as e:  # graceful-degrade: Redis optional, continue without
                    self._rollback_registered_tools(project_id, registered_tools)
                    restore_error = self._to_restore_error(project_id, e)
                    logger.warning(
                        "Skipped restoring project '%s' and deleted stale Redis record (%s): %s",
                        project_id,
                        restore_error.code,
                        restore_error.message,
                    )
                    await self._remove_persisted_registration(project_id)
        except Exception as e:  # graceful-degrade: Redis optional, continue without
            logger.error("Failed to scan Redis for registrations: %s", e)
        finally:
            await store.close()

    async def _restore_project_from_persistence(self, project_id: str) -> bool:
        """Restore one project registration from Redis when hash idempotency needs in-memory state."""
        store = await self._open_registration_store()
        if not store:
            return False
        registered_tools: set[str] = set()
        try:
            payload = await store.load_payload(store.project_key(project_id))
            if payload is None:
                return False
            if project_id in self._project_tools:
                _ = self._deregister_project(project_id)

            from contextunity.router.modules.tools import register_tool

            allowed_tenants = resolve_bundle_allowed_tenants(dict(payload))
            tool_instances = create_tools_from_bundle(dict(payload), project_id=project_id)
            for tool_instance in tool_instances:
                register_tool(
                    tool_instance,
                    allowed_tenants=tuple(allowed_tenants),
                    project_id=project_id,
                )
                registered_tools.add(tool_instance.name)

            graph_map_raw = payload.get("graph")
            graph_map: JsonDict = graph_map_raw if is_json_dict(graph_map_raw) else {}
            if not graph_map:
                self._rollback_registered_tools(project_id, registered_tools)
                return False

            registered_graph_map: dict[str, str] = {}
            callbacks_map: dict[str, list[str]] = {}
            for graph_key, raw_entry in graph_map.items():
                entry = GraphEntry.model_validate(raw_entry)
                _validate_graph_key_field_consistency(graph_key, "id", entry.id)
                _validate_graph_key_field_consistency(graph_key, "name", entry.name)
                if entry.id is None:
                    entry = entry.model_copy(update={"id": graph_key})
                entry = entry.model_copy(update={"name": graph_key})
                g_name = self._register_graph(
                    project_id,
                    entry,
                    available_tools=registered_tools,
                )
                registered_graph_map[graph_key] = g_name
                if entry.router_callbacks:
                    callbacks_map[graph_key] = list(entry.router_callbacks)

            default_graph_id_raw = payload.get("default_graph")
            default_graph_id = (
                default_graph_id_raw if isinstance(default_graph_id_raw, str) else None
            )
            if default_graph_id and default_graph_id in registered_graph_map:
                registered_graph_map["default"] = registered_graph_map[default_graph_id]
            elif len(registered_graph_map) == 1:
                registered_graph_map["default"] = next(iter(registered_graph_map.values()))

            self._project_graphs[project_id] = registered_graph_map
            self._project_router_callbacks[project_id] = callbacks_map
            self._project_configs[project_id] = registered_project_config_from_persisted(
                dict(payload), graph_map
            )
            self._project_tools[project_id] = sorted(registered_tools)
            return True
        except Exception as exc:
            self._rollback_registered_tools(project_id, registered_tools)
            logger.warning(
                "Failed to restore project '%s' from persistence during hash match: %s",
                project_id,
                exc,
            )
            return False
        finally:
            await store.close()


__all__ = ["PersistenceMixin"]
