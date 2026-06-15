"""Registration mixin — RegisterTools, DeregisterTools, graph registration."""

from __future__ import annotations

import secrets
import threading
from typing import TYPE_CHECKING

import pydantic
from contextunity.core import contextunit_pb2, get_contextunit_logger
from contextunity.core.exceptions import (
    ConfigurationError,
    SecurityError,
)
from contextunity.core.sdk.payload import get_str
from contextunity.core.types import ContextUnitPayload, JsonDict, is_json_dict
from grpc.aio import ServicerContext

from contextunity.router.service.decorators import grpc_error_handler
from contextunity.router.service.graph_registration import (
    deregister_project_graphs,
    register_graph_for_project,
)
from contextunity.router.service.helpers import make_response, parse_unit
from contextunity.router.service.mixins.execution.types import (
    ProjectConfigMap,
    ProjectGraphMap,
    ProjectToolMap,
    RouterCallbackMap,
)
from contextunity.router.service.payloads import GraphEntry, RegisterManifestPayload
from contextunity.router.service.registration_auth import get_verified_registration_auth_context
from contextunity.router.service.registration_projection import (
    registered_project_config_from_bundle,
)
from contextunity.router.service.registration_tools import create_tools_from_bundle

logger = get_contextunit_logger(__name__)

ContextUnit = contextunit_pb2.ContextUnit


def _validate_graph_key_field_consistency(
    graph_key: str,
    field_name: str,
    field_value: str | None,
) -> None:
    """Validate that an explicit graph key-derived field matches its map key.

    Auto-population pattern (v1alpha3): if id is None, the caller sets it from
    the key. If id is explicitly provided and differs from the key, it is a
    configuration error — the key IS the id.

    Args:
        graph_key: The YAML map key (e.g. 'cu-rag').
        field_name: GraphEntry field name.
        field_value: The explicit field value from GraphEntry, or None.

    Raises:
        ConfigurationError: If field_value is set and does not match graph_key.
    """
    if field_value is not None and field_value != graph_key:
        raise ConfigurationError(
            message=(
                f"Graph entry {field_name} '{field_value}' does not match its map key "
                f"'{graph_key}'. The map key is the canonical graph id. Either omit the field "
                "(it will be auto-populated) or set it to the same value as the key."
            )
        )


def _validate_graph_id_consistency(graph_key: str, entry_id: str | None) -> None:
    """Backward-compatible wrapper for tests and older internal imports."""
    _validate_graph_key_field_consistency(graph_key, "id", entry_id)


def _strict_graph_map(bundle: ContextUnitPayload) -> JsonDict:
    """Return the registration graph map, failing closed on malformed bundles."""
    graph_map = bundle.get("graph")
    if not is_json_dict(graph_map) or not graph_map:
        raise ConfigurationError(
            message="RegisterManifest bundle missing non-empty 'graph' dictionary"
        )
    return graph_map


def _validated_graph_entries(graph_map: JsonDict) -> dict[str, GraphEntry]:
    """Validate all graph entries before mutating registries."""
    entries: dict[str, GraphEntry] = {}
    for graph_key, raw_entry in graph_map.items():
        try:
            validated = GraphEntry.model_validate(raw_entry)
        except pydantic.ValidationError as exc:
            raise ConfigurationError(message=f"Invalid graph entry '{graph_key}': {exc}") from exc

        _validate_graph_key_field_consistency(graph_key, "id", validated.id)
        _validate_graph_key_field_consistency(graph_key, "name", validated.name)
        if validated.id is None:
            validated = validated.model_copy(update={"id": graph_key})
        entries[graph_key] = validated.model_copy(update={"name": graph_key})
    return entries


def _validate_default_graph(default_graph_id: str, graph_entries: dict[str, GraphEntry]) -> None:
    """Ensure multi-graph bundles declare an unambiguous default graph."""
    if default_graph_id:
        if default_graph_id not in graph_entries:
            raise ConfigurationError(
                message=f"default_graph '{default_graph_id}' is not present in graph map"
            )
        return
    if len(graph_entries) > 1:
        raise ConfigurationError(
            message="RegisterManifest multi-graph bundles must declare a valid default_graph"
        )


class RegistrationMixin:
    """Mixin providing RegisterManifest, and graph management."""

    _project_graphs: ProjectGraphMap = {}
    _project_configs: ProjectConfigMap = {}
    _project_tools: ProjectToolMap = {}
    _project_router_callbacks: RouterCallbackMap = {}
    _stream_secrets: dict[str, str] = {}
    _stream_secrets_lock: threading.Lock = threading.Lock()

    def get_cached_stream_secret(self, project_id: str) -> str | None:
        """Return in-memory stream auth secret for *project_id*, if cached."""
        with self._stream_secrets_lock:
            return self._stream_secrets.get(project_id)

    def put_cached_stream_secret(self, project_id: str, secret: str) -> None:
        """Cache stream auth secret for reconnecting project executors."""
        with self._stream_secrets_lock:
            self._stream_secrets[project_id] = secret

    if TYPE_CHECKING:

        async def _check_manifest_hash(self, project_id: str, current_hash: str) -> bool:
            del project_id, current_hash
            raise NotImplementedError

        async def _persist_registration(self, project_id: str, payload: dict[str, object]) -> None:
            del project_id, payload
            raise NotImplementedError

        async def _save_manifest_hash(self, project_id: str, new_hash: str) -> None:
            del project_id, new_hash
            raise NotImplementedError

        async def _persist_stream_secret(self, project_id: str, stream_secret: str) -> None:
            del project_id, stream_secret
            raise NotImplementedError

        async def _restore_project_from_persistence(self, project_id: str) -> bool:
            del project_id
            raise NotImplementedError

    @grpc_error_handler
    async def RegisterManifest(
        self,
        request: ContextUnit,
        context: ServicerContext[ContextUnit, ContextUnit],
    ) -> ContextUnit:
        """Register project tools and graph via manifest or pre-compiled bundle.

        Accepts either:
          - bundle: pre-compiled by ArtifactGenerator on project side (preferred)
          - manifest: raw ContextUnityProject dict (backward compat — compiled here)

        When bundle includes inline secrets (no-Shield scenario), stores them
        per-tenant in _project_secrets for create_llm() fallback.
        """
        unit = parse_unit(request)

        try:
            params = RegisterManifestPayload.model_validate(unit.payload or {})
        except pydantic.ValidationError as e:
            raise ConfigurationError(
                message=f"Invalid RegisterManifest payload: {e}",
            ) from e

        bundle_raw = params.bundle
        if not bundle_raw:
            raise ConfigurationError(
                message=(
                    "RegisterManifest requires 'bundle' — a pre-compiled registration bundle "
                    "from ArtifactGenerator (contextunity.core.manifest.generators)"
                ),
            )
        bundle: ContextUnitPayload = dict(bundle_raw)
        from contextunity.core.manifest.bundle_hash import compute_registration_bundle_hash

        server_manifest_hash = compute_registration_bundle_hash(dict(bundle_raw))

        project_id = get_str(bundle, "project_id")
        if not project_id:
            raise ConfigurationError(
                message="RegisterManifest bundle missing non-empty 'project_id'"
            )

        from contextunity.core.manifest.tenants import (
            require_token_covers_allowed_tenants,
            resolve_bundle_allowed_tenants,
        )

        allowed_tenants = resolve_bundle_allowed_tenants(bundle)
        bundle["allowed_tenants"] = allowed_tenants

        if "project_secret" in bundle:
            raise SecurityError(
                message=(
                    "RegisterManifest payload must NOT contain 'project_secret'. "
                    "HMAC secrets are resolved from CU_PROJECT_SECRET env var. "
                    "Update your contextunity.core SDK."
                ),
            )

        from contextunity.core.authz import authorize
        from contextunity.core.discovery import (
            get_or_create_project_stream_secret,
            register_project,
            verify_project_owner,
        )

        auth_ctx = await get_verified_registration_auth_context(
            context,
            project_id=project_id,
        )
        require_token_covers_allowed_tenants(
            auth_ctx,
            allowed_tenants=allowed_tenants,
            project_id=project_id,
        )
        decision = authorize(
            auth_ctx,
            registration_project_id=project_id,
            service="router",
            rpc_name="RegisterManifest",
        )
        if decision.denied:
            raise SecurityError(
                f"Registration denied for project '{project_id}': {decision.reason}"
            )

        if not verify_project_owner(project_id):
            raise SecurityError(
                f"Project '{project_id}' is already registered to a different owner. "
                + "Cannot hijack identity."
            )

        is_matched = await self._check_manifest_hash(project_id, server_manifest_hash)
        if is_matched:
            logger.debug(
                "Project '%s' manifest hash matched. Skipping re-registration.", project_id
            )
            from contextunity.router.service.shield_client import get_shield_url

            # Idempotent re-registration: reuse the existing stream
            # secret so active ToolExecutorStream sessions reconnect
            # with the same key. A new manifest that changes the
            # graph structure still goes through the non-hash-match
            # path below and rotates the secret there.
            stream_secret = get_or_create_project_stream_secret(project_id)
            with self._stream_secrets_lock:
                self._stream_secrets[project_id] = stream_secret
            await self._persist_stream_secret(project_id, stream_secret)

            graph_map = self._project_graphs.get(project_id)
            if not isinstance(graph_map, dict):
                _ = await self._restore_project_from_persistence(project_id)
                graph_map = self._project_graphs.get(project_id)
            if not isinstance(graph_map, dict):
                raise ConfigurationError(
                    message=f"Project '{project_id}' has no registered graph map"
                )
            graph_entry = graph_map.get("default", "")

            return make_response(
                payload={
                    "registered_tools": self._project_tools.get(project_id, []),
                    "graph": graph_entry.replace(f"project:{project_id}:", ""),
                    "status": "ok",
                    "hash_matched": True,
                    "stream_secret": stream_secret,
                    "shield_url": get_shield_url(),
                },
                trace_id=str(unit.trace_id),
                security=unit.security,
            )

        # ── Store inline secrets (no-Shield fallback) ──────────────
        inline_secrets_raw = bundle.pop("secrets", None)
        inline_secrets: dict[str, str] | None = None
        if inline_secrets_raw is not None:
            if not is_json_dict(inline_secrets_raw):
                raise SecurityError("Inline registration secrets must be a JSON object")
            inline_secrets = {}
            for key, value in inline_secrets_raw.items():
                if not isinstance(value, str) or not value:
                    raise SecurityError("Inline registration secrets must be non-empty strings")
                inline_secrets[str(key)] = value

        # Validate and instantiate everything before mutating global registries.
        graph_map = _strict_graph_map(bundle)
        graph_entries = _validated_graph_entries(graph_map)
        default_graph_id = get_str(bundle, "default_graph")
        _validate_default_graph(default_graph_id, graph_entries)
        tool_instances = create_tools_from_bundle(bundle, project_id=project_id)

        from contextunity.router.core.registry import graph_registry
        from contextunity.router.modules.tools import (
            get_tool_for_project,
            list_project_tools,
            register_tool,
        )

        had_old_tools = project_id in self._project_tools
        had_old_graphs = project_id in self._project_graphs
        had_old_config = project_id in self._project_configs
        had_old_callbacks = project_id in self._project_router_callbacks
        old_tool_names = list(self._project_tools.get(project_id, []))
        old_project_tool_names = set(list_project_tools(project_id))
        old_tool_instances = [
            tool
            for name in old_tool_names
            if name in old_project_tool_names
            if (tool := get_tool_for_project(project_id, name)) is not None
        ]
        old_graph_map = self._project_graphs.get(project_id, {})
        old_project_config = self._project_configs.get(project_id, {})
        old_callbacks = self._project_router_callbacks.get(project_id, {})
        old_graph_factories = {
            name: graph_registry.get(name)
            for name in set(old_graph_map.values())
            if graph_registry.has(name)
        }

        if project_id in self._project_tools:
            _ = self._deregister_project(project_id)
        _ = self._project_configs.pop(project_id, None)

        registered_tools: list[str] = []
        graph_name = ""

        try:
            for tool_instance in tool_instances:
                register_tool(
                    tool_instance,
                    allowed_tenants=tuple(allowed_tenants),
                    project_id=project_id,
                )
                registered_tools.append(tool_instance.name)

            self._project_tools[project_id] = registered_tools

            registered_graph_map: dict[str, str] = {}
            callbacks_map: dict[str, list[str]] = {}
            for graph_key, validated in graph_entries.items():
                g_name = self._register_graph(
                    project_id,
                    validated,
                    available_tools=set(registered_tools),
                )
                registered_graph_map[graph_key] = g_name

                if validated.router_callbacks:
                    callbacks_map[graph_key] = list(validated.router_callbacks)

            if default_graph_id:
                registered_graph_map["default"] = registered_graph_map[default_graph_id]
            else:
                registered_graph_map["default"] = next(iter(registered_graph_map.values()))

            self._project_graphs[project_id] = registered_graph_map
            self._project_router_callbacks[project_id] = callbacks_map
            self._project_configs[project_id] = registered_project_config_from_bundle(
                bundle, graph_map
            )
            graph_name = registered_graph_map["default"]
        except Exception:
            _ = self._deregister_project(project_id)
            _ = self._project_configs.pop(project_id, None)
            for tool_instance in old_tool_instances:
                register_tool(tool_instance, project_id=project_id)
            for registry_name, graph_factory in old_graph_factories.items():
                graph_registry.register(registry_name, graph_factory, overwrite=True)
            if had_old_tools:
                self._project_tools[project_id] = old_tool_names
            if had_old_graphs:
                self._project_graphs[project_id] = old_graph_map
            if had_old_config:
                self._project_configs[project_id] = old_project_config
            if had_old_callbacks:
                self._project_router_callbacks[project_id] = old_callbacks
            raise

        persisted_bundle = dict(bundle)
        persisted_bundle["__registration_hash"] = server_manifest_hash
        await self._persist_registration(project_id, persisted_bundle)

        # Update registered tools list in redis and persist inline secrets (open-source scale-safe)
        from contextunity.core.config import get_core_config as _get_shared_config

        # Always pass the current project_secret so stale Redis entries
        # (e.g. from a previous Shield/enterprise session) get overwritten.
        _current_secret = _get_shared_config().security.project_secret or None

        stream_secret = secrets.token_urlsafe(32)
        # First-time (or non-hash-match) registration. Mint a fresh
        # stream secret. Active sessions cannot survive this — the
        # executor must re-handshake with the new secret, which is the
        # intended behavior when the manifest graph structure changes.
        with self._stream_secrets_lock:
            self._stream_secrets[project_id] = stream_secret

        await self._persist_stream_secret(project_id, stream_secret)

        # Legacy discovery store: HMAC secret + inline API keys until SSOT migration completes.
        _ = register_project(
            project_id,
            tools=registered_tools,
            api_keys=inline_secrets,
            project_secret=_current_secret,
        )

        from contextunity.router.service.shield_client import get_shield_url

        response_payload = {
            "registered_tools": registered_tools,
            "graph": graph_name.replace(f"project:{project_id}:", ""),
            "status": "ok",
            "hash_matched": False,
            "stream_secret": stream_secret,
            "shield_url": get_shield_url(),
        }

        return make_response(
            payload=response_payload,
            trace_id=str(unit.trace_id),
            security=unit.security,
        )

    def _register_graph(
        self,
        project_id: str,
        graph_config: GraphEntry,
        *,
        available_tools: set[str] | None = None,
    ) -> str:
        """Register a graph from config."""
        return register_graph_for_project(
            project_id,
            graph_config,
            available_tools=available_tools,
        )

    def _deregister_project(self, project_id: str) -> list[str]:
        """Remove all tools and graph for a project."""
        return deregister_project_graphs(
            project_tools=self._project_tools,
            project_graphs=self._project_graphs,
            project_router_callbacks=self._project_router_callbacks,
            project_configs=self._project_configs,
            project_id=project_id,
        )


__all__ = ["RegistrationMixin", "_validate_graph_id_consistency"]
