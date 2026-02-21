"""Registration mixin — RegisterTools, DeregisterTools, graph registration."""

from __future__ import annotations

import secrets

from contextcore import get_context_unit_logger

from contextrouter.core import get_core_config
from contextrouter.service.decorators import grpc_error_handler
from contextrouter.service.helpers import make_response, parse_unit
from contextrouter.service.payloads import DeregisterToolsPayload, RegisterToolsPayload
from contextrouter.service.tool_factory import create_tool_from_config

logger = get_context_unit_logger(__name__)


class RegistrationMixin:
    """Mixin providing RegisterTools, DeregisterTools, and graph management."""

    @grpc_error_handler
    async def RegisterTools(self, request, context):
        """Register project tools and graph in Router.

        Creates real BaseTool instances in Router's process.
        DB URLs are stored securely in closure scope.

        Security: Requires 'tools:register' write scope.
        """
        unit = parse_unit(request)

        # Accept both generic "tools:register" and project-specific "tools:register:{id}"
        unit_write = set(unit.security.write or []) if unit.security else set()
        has_scope = any(
            w == "tools:register" or w.startswith("tools:register:") for w in unit_write
        )
        if not has_scope:
            config = get_core_config()
            if config.security.enabled:
                raise PermissionError("Missing 'tools:register' write scope")

        try:
            params = RegisterToolsPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid RegisterTools payload: {e}") from e

        # ── VULN-5: Validate graph config (prompt injection defense) ──
        if params.graph and hasattr(params.graph, "config"):
            cfg = params.graph.config if isinstance(params.graph.config, dict) else {}
            prompt = cfg.get("planner_prompt", "")
            if isinstance(prompt, str) and len(prompt) > 10_000:
                raise ValueError(
                    f"planner_prompt exceeds maximum length (10,000 chars, "
                    f"got {len(prompt)}). Reduce prompt size to prevent "
                    f"context window exhaustion."
                )

        # ── Three-layer registration security ──────────────
        # Layer A: token.allowed_tenants — is this tenant mine?
        # Layer B: tools:register:{project_id} — granular permission
        # Layer C: Redis project registry — server-side ownership
        config = get_core_config()
        if config.security.enabled:
            from contextcore import extract_token_from_grpc_metadata
            from contextcore.permissions import has_registration_access

            token = extract_token_from_grpc_metadata(context)
            if token is not None:
                # Layer A: Tenant binding
                allowed = getattr(token, "allowed_tenants", ()) or ()
                if allowed and params.project_id not in allowed:
                    raise PermissionError(
                        f"Tenant ownership violation: cannot register tools for "
                        f"project '{params.project_id}'. Token allows: {list(allowed)}"
                    )

                # Layer B: Granular registration permission
                perms = getattr(token, "permissions", ()) or ()
                if not has_registration_access(perms, params.project_id):
                    raise PermissionError(
                        f"Missing registration permission for project "
                        f"'{params.project_id}'. Need 'tools:register' or "
                        f"'tools:register:{params.project_id}'"
                    )

                # Layer C: Redis project registry
                # Use project_id as canonical owner identity
                owner_tenant = params.project_id
                from contextcore.discovery import register_project, verify_project_owner

                if not verify_project_owner(params.project_id, owner_tenant):
                    raise PermissionError(
                        f"Project '{params.project_id}' is already registered by "
                        f"a different owner. Cannot hijack."
                    )
                register_project(
                    params.project_id,
                    owner_tenant,
                    tools=[t.name for t in params.tools],
                )
            else:
                # Fail-closed: no token → reject registration
                raise PermissionError(
                    "No token in gRPC metadata for RegisterTools — "
                    f"cannot verify tenant ownership for project '{params.project_id}'. "
                    "Provide a ContextToken with allowed_tenants and "
                    "tools:register permission."
                )

        # Idempotent re-registration
        if params.project_id in self._project_tools:
            self._deregister_project(params.project_id)

        registered_tools: list[str] = []

        for tool_def in params.tools:
            try:
                tools = create_tool_from_config(
                    name=tool_def.name,
                    tool_type=tool_def.type,
                    description=tool_def.description,
                    config=tool_def.config,
                )
                from contextrouter.modules.tools import register_tool

                for tool_instance in tools:
                    register_tool(tool_instance, tenant=params.project_id)
                    registered_tools.append(tool_instance.name)
                    logger.info(
                        "Registered tool '%s' for project '%s'",
                        tool_instance.name,
                        params.project_id,
                    )
            except Exception as e:
                logger.error(
                    "Failed to create tool '%s' for project '%s': %s",
                    tool_def.name,
                    params.project_id,
                    e,
                )
                raise ValueError(f"Failed to create tool '{tool_def.name}': {e}") from e

        self._project_tools[params.project_id] = registered_tools

        graph_name = None
        if params.graph:
            graph_name = self._register_graph(params.project_id, params.graph)

        await self._persist_registration(params.project_id, unit.payload)

        # ── Per-registration stream secret (one-time use) ──────────
        # Generated here, stored in Router memory, returned to project.
        # Consumed (deleted) on first stream connect → one-time use.
        # Enterprise deployments can additionally use Shield for audit.
        stream_secret = secrets.token_urlsafe(32)
        with self._stream_secrets_lock:
            self._stream_secrets[params.project_id] = stream_secret

        logger.info(
            "Project '%s' registered: %d tools, graph=%s",
            params.project_id,
            len(registered_tools),
            graph_name or "none",
        )

        response_payload = {
            "registered_tools": registered_tools,
            "graph": graph_name,
            "status": "ok",
            "stream_secret": stream_secret,
        }

        return make_response(
            payload=response_payload,
            trace_id=str(unit.trace_id),
            provenance=list(unit.provenance) + ["router:register_tools"],
            security=unit.security,
        )

    @grpc_error_handler
    async def DeregisterTools(self, request, context):
        """Deregister project tools and graph.

        Security: Enforces tenant ownership — only the project owner
        can deregister their own tools.
        """
        unit = parse_unit(request)

        # Accept both generic and project-specific scopes
        unit_write = set(unit.security.write or []) if unit.security else set()
        has_scope = any(
            w == "tools:register" or w.startswith("tools:register:") for w in unit_write
        )
        if not has_scope:
            config = get_core_config()
            if config.security.enabled:
                raise PermissionError("Missing 'tools:register' write scope")

        try:
            params = DeregisterToolsPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid DeregisterTools payload: {e}") from e

        # ── Ownership check (mirrors RegisterTools layers A+B) ──
        config = get_core_config()
        if config.security.enabled:
            from contextcore import extract_token_from_grpc_metadata
            from contextcore.permissions import has_registration_access

            token = extract_token_from_grpc_metadata(context)
            if token is None:
                raise PermissionError(
                    "No token in gRPC metadata for DeregisterTools — "
                    f"cannot verify ownership for project '{params.project_id}'."
                )

            # Layer A: Tenant binding
            allowed = getattr(token, "allowed_tenants", ()) or ()
            if allowed and params.project_id not in allowed:
                raise PermissionError(
                    f"Tenant ownership violation: cannot deregister tools for "
                    f"project '{params.project_id}'. Token allows: {list(allowed)}"
                )

            # Layer B: Granular permission
            perms = getattr(token, "permissions", ()) or ()
            if not has_registration_access(perms, params.project_id):
                raise PermissionError(
                    f"Missing registration permission for deregistering "
                    f"project '{params.project_id}'."
                )

        deregistered = self._deregister_project(params.project_id)
        await self._remove_persisted_registration(params.project_id)

        # Clean up stream secret (thread-safe)
        with self._stream_secrets_lock:
            self._stream_secrets.pop(params.project_id, None)

        return make_response(
            payload={"deregistered": deregistered, "status": "ok"},
            trace_id=str(unit.trace_id),
            provenance=list(unit.provenance) + ["router:deregister_tools"],
            security=unit.security,
        )

    def _register_graph(self, project_id: str, graph_config: object) -> str:
        """Register a graph from config.

        Currently supports template-based graphs (Option A).
        Declarative graphs (Option B) are a future extension.
        """
        from contextrouter.core.registry import graph_registry

        name = graph_config.name

        if graph_config.template:
            template = graph_config.template
            config = graph_config.config

            if template == "sql_analytics":
                builder = self._build_sql_analytics_graph_factory(config)
            elif template == "dispatcher":
                from contextrouter.cortex.graphs.dispatcher_agent import (
                    build_dispatcher_graph,
                )

                builder = build_dispatcher_graph
            elif template == "rag_retrieval":
                from contextrouter.cortex.graphs.rag_retrieval.graph import (
                    build_graph as build_rag_graph,
                )

                builder = build_rag_graph
            else:
                raise ValueError(
                    f"Unknown graph template: {template}. "
                    f"Available: sql_analytics, dispatcher, rag_retrieval"
                )

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
            return name

        elif graph_config.nodes and graph_config.edges:
            raise ValueError(
                "Declarative graph registration is not yet implemented. "
                "Use template-based registration (Option A)."
            )
        else:
            raise ValueError("Graph config must have either 'template' or 'nodes'+'edges'")

    def _build_sql_analytics_graph_factory(self, config: dict):
        """Create a graph builder function for sql_analytics template."""

        def builder():
            from contextrouter.cortex.graphs.sql_analytics.builder import (
                build_sql_analytics_graph,
            )

            return build_sql_analytics_graph(config)

        return builder

    def _deregister_project(self, project_id: str) -> list[str]:
        """Remove all tools and graph for a project."""
        from contextrouter.modules.tools import deregister_tool

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
