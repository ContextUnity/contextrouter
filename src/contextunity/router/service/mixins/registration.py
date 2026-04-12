"""Registration mixin — RegisterTools, DeregisterTools, graph registration."""

from __future__ import annotations

import secrets

from contextunity.core import get_contextunit_logger

from contextunity.router.service.decorators import grpc_error_handler
from contextunity.router.service.helpers import make_response, parse_unit
from contextunity.router.service.tool_factory import create_tool_from_config

logger = get_contextunit_logger(__name__)

# Module-level store for inline API keys (no-Shield fallback).
# Populated by RegisterManifest when bundle includes inline secrets.
# Accessed by ModelRegistry.create_llm() as Shield fallback.
# Dict: project_id → {provider: api_key}
_project_secrets: dict[str, dict[str, str]] = {}


def _extract_registration_token_string(context) -> str:
    """Extract bearer token from gRPC metadata for registration RPCs."""
    metadata = dict(context.invocation_metadata() or [])

    auth_header = str(metadata.get("authorization", "")).strip()
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    token_str = str(metadata.get("x-context-token", "")).strip()
    if token_str:
        return token_str

    return ""


async def _build_registration_verifier(
    *,
    token_str: str,
    project_id: str,
):
    """Build a verifier backend for registration/auth bootstrap.

    Resolution order:
    1. Redis: stored project_secret or public_key (from previous registration)
    2. Shield: fetch Ed25519 public key (Enterprise mode)
    3. Env: CU_PROJECT_SECRET from SharedConfig (HMAC single-project mode)
    """
    parts = token_str.rsplit(".", 2)
    if len(parts) != 3:
        raise PermissionError("Registration token must use composite kid wire format")

    kid = parts[0]
    if ":" not in kid:
        raise PermissionError(
            "Registration token must use composite kid format '<project>:<key-version>'"
        )

    caller_project_id, key_version = kid.split(":", 1)
    if caller_project_id != project_id:
        raise PermissionError(
            f"Registration token project mismatch: token project '{caller_project_id}' "
            f"cannot register project '{project_id}'"
        )

    from contextunity.core.discovery import get_project_key

    key_data = get_project_key(project_id) or {}
    stored_secret = key_data.get("project_secret")

    if "session" in key_version:
        public_key_b64 = key_data.get("public_key_b64")
        if not public_key_b64:
            from contextunity.router.service.shield_client import _get_shield_url

            shield_url = _get_shield_url()
            if not shield_url:
                raise PermissionError(
                    f"No public key cached for project '{project_id}' and Shield is not configured"
                )

            from contextunity.core.token_utils import fetch_project_public_key_async

            public_key_b64, returned_kid = await fetch_project_public_key_async(
                project_id,
                kid,
                shield_url,
                provenance="router:register_manifest:fetch_public_key",
            )

            from contextunity.core.discovery import update_project_public_key

            update_project_public_key(project_id, public_key_b64, returned_kid)

        try:
            from contextunity.core.ed25519 import Ed25519Backend
        except ImportError as exc:
            raise PermissionError("contextunity.core.ed25519 failed to import") from exc

        return Ed25519Backend(public_key_b64=public_key_b64, kid=kid)

    # HMAC mode: try stored secret first, then env fallback
    if stored_secret:
        from contextunity.core.signing import HmacBackend

        return HmacBackend(project_id, stored_secret)

    # Env fallback: CU_PROJECT_SECRET from SharedConfig
    from contextunity.core.config import get_core_config

    env_secret = get_core_config().security.project_secret
    if env_secret:
        from contextunity.core.signing import HmacBackend

        return HmacBackend(project_id, env_secret)

    raise PermissionError(
        f"No HMAC secret available for project '{project_id}'. "
        f"Set CU_PROJECT_SECRET env var or use Shield mode (Ed25519)."
    )


async def _get_verified_registration_auth_context(
    context,
    *,
    project_id: str,
):
    """Verify registration token and return canonical auth context."""
    from contextunity.core.authz.context import VerifiedAuthContext
    from contextunity.core.token_utils import verify_token_string

    token_str = _extract_registration_token_string(context)
    if not token_str:
        raise PermissionError("Missing registration token in gRPC metadata")

    backend = await _build_registration_verifier(
        token_str=token_str,
        project_id=project_id,
    )
    token = verify_token_string(token_str, backend)
    if token is None:
        raise PermissionError("Registration token cryptographic verification failed")
    if token.is_expired():
        raise PermissionError(f"Registration token expired for project '{project_id}'")

    return VerifiedAuthContext.from_token(token, token_str, project_id=project_id)


class RegistrationMixin:
    """Mixin providing RegisterManifest, and graph management."""

    @grpc_error_handler
    async def RegisterManifest(self, request, context):
        """Register project tools and graph via manifest or pre-compiled bundle.

        Accepts either:
          - bundle: pre-compiled by ArtifactGenerator on project side (preferred)
          - manifest: raw ContextUnityProject dict (backward compat — compiled here)

        When bundle includes inline secrets (no-Shield scenario), stores them
        per-tenant in _project_secrets for create_llm() fallback.
        """
        unit = parse_unit(request)

        from contextunity.router.service.payloads import RegisterManifestPayload

        try:
            params = RegisterManifestPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid RegisterManifest payload: {e}") from e

        bundle = params.bundle
        if not bundle:
            raise ValueError(
                "RegisterManifest requires 'bundle' — a pre-compiled registration bundle "
                "from ArtifactGenerator (cu.core.manifest.generators)"
            )

        project_id = bundle.get("project_id", "")
        tenant_id = bundle.get("tenant_id", project_id)

        if "project_secret" in bundle:
            raise ValueError(
                "RegisterManifest payload must NOT contain 'project_secret'. "
                "HMAC secrets are resolved from CU_PROJECT_SECRET env var. "
                "Update your cu.core SDK."
            )

        from contextunity.core.authz import authorize
        from contextunity.core.discovery import register_project, verify_project_owner

        auth_ctx = await _get_verified_registration_auth_context(
            context,
            project_id=project_id,
        )
        decision = authorize(
            auth_ctx,
            registration_project_id=project_id,
            tenant_id=tenant_id,
            service="router",
            rpc_name="RegisterManifest",
        )
        if decision.denied:
            raise PermissionError(
                f"Registration denied for project '{project_id}': {decision.reason}"
            )

        if not verify_project_owner(project_id, tenant_id):
            raise PermissionError(
                f"Project '{project_id}' is already registered to "
                f"a different tenant. Cannot hijack identity."
            )
        register_project(project_id, tenant_id, tools=[])

        # Hash Idempotency check
        if params.hash:
            is_matched = await self._check_manifest_hash(project_id, params.hash)
            if is_matched:
                logger.debug(
                    "Project '%s' manifest hash matched. Skipping re-registration.", project_id
                )
                from contextunity.router.service.shield_client import _get_shield_url

                stream_secret = self._stream_secrets.get(project_id)
                return make_response(
                    payload={
                        "registered_tools": self._project_tools.get(project_id, []),
                        "graph": self._project_graphs.get(project_id, "").replace(
                            f"project:{project_id}:", ""
                        ),
                        "status": "ok",
                        "hash_matched": True,
                        "stream_secret": stream_secret,
                        "shield_url": _get_shield_url(),
                    },
                    trace_id=str(unit.trace_id),
                    security=unit.security,
                )

        # ── Store inline secrets (no-Shield fallback) ──────────────
        inline_secrets = bundle.pop("secrets", None)
        if inline_secrets:
            _project_secrets[project_id] = inline_secrets
            logger.info(
                "Stored %d inline API key(s) for project '%s' (no-Shield mode)",
                len(inline_secrets),
                project_id,
            )

        # De-register existing tools
        if project_id in self._project_tools:
            self._deregister_project(project_id)

        registered_tools: list[str] = []

        from contextunity.router.service.payloads import GraphConfig, ToolConfig

        for tool_dict in bundle.get("tools", []):
            try:
                tool_name = tool_dict.get("name")
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
            except Exception as e:
                logger.error(
                    "Failed to create tool '%s' for project '%s': %s", tool_name, project_id, e
                )
                raise ValueError(f"Failed to create tool '{tool_name}': {e}") from e

        self._project_tools[project_id] = registered_tools

        graph_name = None
        graph_dict = bundle.get("graph")
        if graph_dict:
            graph_config = GraphConfig(
                name=graph_dict["id"],
                template=graph_dict.get("template", "custom"),
                nodes=graph_dict.get("nodes", []),
                edges=graph_dict.get("edges", []),
                config=graph_dict.get("config", {}),
            )
            graph_name = self._register_graph(project_id, graph_config)

            # Enrich the stored config with manifest-level metadata required for execution (policies, nodes)
            if project_id not in self._project_configs:
                self._project_configs[project_id] = {}
            self._project_configs[project_id]["nodes"] = graph_dict.get("nodes", [])
            self._project_configs[project_id]["policy"] = bundle.get("policy", {})
            self._project_configs[project_id]["tools"] = bundle.get("tools", [])

        await self._persist_registration(project_id, bundle)

        if params.hash:
            await self._save_manifest_hash(project_id, params.hash)

        # Update registered tools list in redis
        if len(registered_tools) > 0:
            from contextunity.core.discovery import register_project

            register_project(project_id, tenant_id, tools=registered_tools)

        stream_secret = secrets.token_urlsafe(32)
        with self._stream_secrets_lock:
            self._stream_secrets[project_id] = stream_secret

        from contextunity.router.service.shield_client import _get_shield_url

        response_payload = {
            "registered_tools": registered_tools,
            "graph": graph_name,
            "status": "ok",
            "hash_matched": False,
            "stream_secret": stream_secret,
            "shield_url": _get_shield_url(),
        }

        return make_response(
            payload=response_payload,
            trace_id=str(unit.trace_id),
            security=unit.security,
        )

    def _register_graph(self, project_id: str, graph_config: object) -> str:
        """Register a graph from config.

        v1alpha runtime supports template-based graphs only.
        """
        from contextunity.router.core.registry import graph_registry

        # Namespace the graph name to prevent replacing core system graphs
        original_name = graph_config.name
        name = f"project:{project_id}:{original_name}"

        if graph_config.template and graph_config.template != "custom":
            template = graph_config.template
            config = graph_config.config

            if template == "sql_analytics":
                builder = self._build_sql_analytics_graph_factory(config)
            elif template == "gardener":
                from contextunity.router.cortex.graphs.commerce.gardener.graph import (
                    build_gardener_graph,
                )

                def builder(_config=config):
                    return build_gardener_graph(_config)
            elif template == "dispatcher":
                from contextunity.router.cortex.graphs.dispatcher_agent import (
                    build_dispatcher_graph,
                )

                builder = build_dispatcher_graph
            elif template == "rag_retrieval":
                from contextunity.router.cortex.graphs.rag_retrieval.graph import (
                    build_graph as build_rag_graph,
                )

                builder = build_rag_graph
            elif template == "news_engine":
                from contextunity.router.cortex.graphs.news_engine.graph import (
                    build_news_engine_graph,
                )

                builder = build_news_engine_graph
            else:
                raise ValueError(
                    f"Unknown graph template: {template}. "
                    f"Available: sql_analytics, gardener, dispatcher, rag_retrieval, news_engine"
                )

            # Prevent overwriting since it's already properly namespaced and deregistered first
            graph_registry.register(name, builder, overwrite=True)
            self._project_graphs[project_id] = name
            if isinstance(config, dict):
                self._project_configs[project_id] = config
            logger.info(
                "Registered graph '%s' (template=%s) for project '%s'",
                name,
                template,
                project_id,
            )
            return original_name

        elif graph_config.template == "custom":
            name = f"project:{project_id}:{graph_config.name}"

            def builder():
                from contextunity.router.cortex.graphs.custom.builder import build_custom_graph

                return build_custom_graph(
                    graph_config.nodes, graph_config.edges, graph_config.config
                )

            from contextunity.router.core.registry import graph_registry

            graph_registry.register(name, builder, overwrite=True)
            self._project_graphs[project_id] = name
            if isinstance(graph_config.config, dict):
                self._project_configs[project_id] = graph_config.config

            logger.info("Registered custom graph '%s' for project '%s'", name, project_id)
            return graph_config.name

        else:
            raise ValueError("Graph config must have either template=<name> or template='custom'")

    def _build_sql_analytics_graph_factory(self, config: dict):
        """Create a graph builder function for sql_analytics template."""

        def builder():
            from contextunity.router.cortex.graphs.sql_analytics.builder import (
                build_sql_analytics_graph,
            )

            return build_sql_analytics_graph(config)

        return builder

    def _deregister_project(self, project_id: str) -> list[str]:
        """Remove all tools and graph for a project."""
        from contextunity.router.modules.tools import deregister_tool

        deregistered: list[str] = []

        tool_names = self._project_tools.pop(project_id, [])
        for name in tool_names:
            if deregister_tool(name):
                deregistered.append(name)

        graph_name = self._project_graphs.pop(project_id, None)
        if graph_name:
            logger.info("Deregistered graph '%s' (project '%s')", graph_name, project_id)
            deregistered.append(f"graph:{graph_name}")

        return deregistered


__all__ = ["RegistrationMixin"]
