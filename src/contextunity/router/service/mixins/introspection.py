"""Introspection mixin — sanitized manifest query for observability."""

from __future__ import annotations

from contextunity.core import contextunit_pb2, get_contextunit_logger
from contextunity.core.permissions.access import has_introspection_access
from contextunity.core.types import is_object_dict
from grpc.aio import ServicerContext

from contextunity.router.service.helpers import make_response, parse_unit
from contextunity.router.service.introspection_contract import (
    IntrospectionGraphJsonWire,
    IntrospectionPolicyWire,
    IntrospectionProjectJsonWire,
    build_introspection_project_json,
    build_introspection_tools_from_config,
    introspection_registrations_payload,
    sanitize_introspection_graph,
    sanitize_introspection_policy,
    sanitize_introspection_services,
)
from contextunity.router.service.mixins.execution.types import (
    ProjectConfigMap,
    ProjectGraphMap,
    ProjectToolMap,
)
from contextunity.router.service.security import validate_introspection_access

logger = get_contextunit_logger(__name__)

ContextUnit = contextunit_pb2.ContextUnit


class IntrospectionMixin:
    """Mixin providing IntrospectRegistrations RPC."""

    _project_graphs: ProjectGraphMap = {}
    _project_configs: ProjectConfigMap = {}
    _project_tools: ProjectToolMap = {}

    async def IntrospectRegistrations(
        self,
        request: ContextUnit,
        context: ServicerContext[ContextUnit, ContextUnit],
    ) -> ContextUnit:
        """Return sanitized manifest data for registered projects.

        All data comes from in-memory state populated during
        RegisterManifest — no Redis round-trip needed.

        Project visibility is gated by **permissions** (``router:introspect:{pid}``,
        ``tools:register:{pid}``, or ``admin:all``), not by ``allowed_tenants``.
        """
        unit = parse_unit(request)
        requested_raw = (unit.payload or {}).get("project_id")
        requested_project = requested_raw if isinstance(requested_raw, str) else None

        token = validate_introspection_access(
            unit,
            context,
            project_id=requested_project,
        )

        if requested_project:
            project_ids: list[str] = [requested_project]
        else:
            project_ids = [
                pid
                for pid in self._project_tools
                if has_introspection_access(token.permissions, pid)
            ]

        projects: list[IntrospectionProjectJsonWire] = []
        for pid in project_ids:
            if pid not in self._project_tools:
                continue

            config = self._project_configs.get(pid, {})
            tools = build_introspection_tools_from_config(config.get("tools", []))

            raw_graph_map = config.get("graph", {})
            graphs: dict[str, IntrospectionGraphJsonWire] = {}
            default_graph: str | None = None
            project_graph_names = self._project_graphs.get(pid, {})
            if "default" in project_graph_names:
                for gkey, compiled in project_graph_names.items():
                    if gkey != "default" and compiled == project_graph_names["default"]:
                        default_graph = gkey
                        break

            if is_object_dict(raw_graph_map):
                for gkey, gdata_obj in raw_graph_map.items():
                    if not is_object_dict(gdata_obj):
                        continue
                    sanitized = sanitize_introspection_graph(gkey, gdata_obj)
                    if gkey == default_graph:
                        sanitized["default"] = True
                    graphs[gkey] = sanitized

            raw_policy = config.get("policy", {})
            policy_wire: IntrospectionPolicyWire = (
                sanitize_introspection_policy(raw_policy) if is_object_dict(raw_policy) else {}
            )
            services_wire = sanitize_introspection_services(config.get("services", {}))
            projects.append(
                build_introspection_project_json(
                    project_id=pid,
                    tools=tools,
                    graphs=graphs,
                    services=services_wire,
                    policy=policy_wire,
                )
            )

        return make_response(
            payload=introspection_registrations_payload(projects),
            trace_id=str(unit.trace_id),
            security=unit.security,
        )


__all__ = ["IntrospectionMixin"]
