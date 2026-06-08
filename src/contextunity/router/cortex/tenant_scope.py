"""Runtime tenant scope resolution for secure nodes and tools."""

from __future__ import annotations

from contextunity.core import ContextToken
from contextunity.core.manifest.tenants import (
    parse_allowed_tenants_field,
    resolve_effective_allowed_tenants,
)
from contextunity.core.types import is_json_dict

from contextunity.router.cortex.compiler.types import CompilerNodeSpec
from contextunity.router.cortex.config_resolution import get_node_config, metadata_project_config
from contextunity.router.cortex.types import GraphState, RegisteredProjectConfig


def _project_tenants(project_config: RegisteredProjectConfig) -> list[str]:
    allowed_raw = project_config.get("allowed_tenants")
    parsed = parse_allowed_tenants_field(allowed_raw)
    if parsed:
        return parsed
    project_id = project_config.get("project_id")
    if isinstance(project_id, str) and project_id:
        return [project_id]
    return []


def resolve_node_effective_tenants(
    state: GraphState,
    node_name: str,
    *,
    token: ContextToken | None = None,
    node_spec: CompilerNodeSpec | None = None,
) -> tuple[str, ...]:
    """Resolve effective tenant scope for a graph node at execution time."""
    project_config = metadata_project_config(state)
    project_tenants = _project_tenants(project_config)
    if not project_tenants:
        raise ValueError(f"Node '{node_name}' missing project allowed_tenants in runtime config")

    node_cfg = node_spec or get_node_config(project_config, node_name)
    graph_key_raw = node_cfg.get("graph_key")
    graph_key = graph_key_raw if isinstance(graph_key_raw, str) else None

    graph_tenants: list[str] | None = None
    if graph_key:
        graph_map = project_config.get("graph")
        if is_json_dict(graph_map):
            graph_entry = graph_map.get(graph_key)
            if is_json_dict(graph_entry):
                graph_tenants = parse_allowed_tenants_field(graph_entry.get("allowed_tenants"))

    node_tenants = parse_allowed_tenants_field(node_cfg.get("allowed_tenants"))

    resolved_token: ContextToken | None = token
    if resolved_token is None:
        injected = state.get("__token__")
        if injected is not None:
            resolved_token = injected
        else:
            resolved_token = state["access_token"]

    token_tenants = resolved_token.allowed_tenants
    return resolve_effective_allowed_tenants(
        project_tenants=project_tenants,
        graph_tenants=graph_tenants,
        node_tenants=node_tenants,
        token_tenants=token_tenants,
        token_is_admin=resolved_token.has_permission("admin:all"),
    )


__all__ = ["resolve_node_effective_tenants"]
