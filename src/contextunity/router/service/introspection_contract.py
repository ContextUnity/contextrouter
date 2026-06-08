"""Shared wire contract and sanitizers for ``IntrospectRegistrations``.

**Ownership:** router defines and emits the RPC shape; **clients** (CLI today, any
future SDK/UI) import these TypedDicts and must not fork parallel trees in core/CLI.

Boundary layers (see ``docs/architecture/type-boundaries.md``):

* **Json wire (L2)** — ``*JsonWire``: keys as on the wire (edge key ``from``). Built by
  router ``sanitize_*`` / ``build_introspection_project_json`` in ``IntrospectionMixin``.
* **Typed view (L4)** — ``Introspection*Wire`` (no ``Json`` suffix): same contract after
  client parse (CLI: ``contextunity.cli.introspection_client.parse_introspection_project``;
  edge key ``from_`` because Python cannot use ``from`` as a TypedDict field name).
"""

from __future__ import annotations

from typing import TypedDict

from contextunity.core.types import is_object_dict, is_object_list

# ---- L4 typed view (router contract; client/CLI after parse) -----------------


class IntrospectionServiceWire(TypedDict, total=False):
    """Service toggles in introspection output."""

    enabled: bool


class IntrospectionToolWire(TypedDict, total=False):
    """Tool entry in introspection output."""

    name: str
    type: str


class IntrospectionGraphNodeWire(TypedDict, total=False):
    """Graph node in introspection output."""

    name: str
    type: str
    model: str
    pii_masking: bool
    tool_binding: str | list[str]
    tools: list[str]


class IntrospectionGraphEdgeWire(TypedDict, total=False):
    """Graph edge in client typed view (CLI/TUI); JSON wire uses key ``from``."""

    from_: str
    to: str
    condition_key: str
    condition_map: dict[str, str]


class IntrospectionGraphWire(TypedDict, total=False):
    """Single graph in client typed view (CLI/TUI); router contract, not CLI-only."""

    name: str
    default: bool
    source: str
    nodes: list[IntrospectionGraphNodeWire]
    edges: list[IntrospectionGraphEdgeWire]


class IntrospectionPolicyWire(TypedDict, total=False):
    """Sanitized policy section from introspection."""

    default_llm: str
    fallback_llms: list[str]
    default_embeddings: str
    langfuse_tracing: bool


class IntrospectionProjectWire(TypedDict, total=False):
    """One project in client typed view after parse (CLI implements parse today)."""

    project_id: str
    services: dict[str, IntrospectionServiceWire]
    tools: list[IntrospectionToolWire]
    graphs: dict[str, IntrospectionGraphWire]
    policy: IntrospectionPolicyWire


# ---- L2 JSON wire (router emits; client/CLI reads, then may parse to L4) -----


IntrospectionGraphEdgeJsonWire = TypedDict(
    "IntrospectionGraphEdgeJsonWire",
    {
        "from": str,
        "to": str,
        "condition_key": str,
        "condition_map": dict[str, str],
    },
    total=False,
)


class IntrospectionGraphJsonWire(TypedDict, total=False):
    """Single graph as emitted on the introspection JSON wire."""

    name: str
    default: bool
    source: str
    nodes: list[IntrospectionGraphNodeWire]
    edges: list[IntrospectionGraphEdgeJsonWire]


class IntrospectionProjectJsonWire(TypedDict, total=False):
    """One project as emitted by router ``IntrospectionMixin`` (L2, pre-client parse)."""

    project_id: str
    services: dict[str, IntrospectionServiceWire]
    tools: list[IntrospectionToolWire]
    graphs: dict[str, IntrospectionGraphJsonWire]
    policy: IntrospectionPolicyWire


def _string_dict(value: object) -> dict[str, str]:
    if not is_object_dict(value):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        if isinstance(raw, str):
            out[key] = raw
    return out


def _string_list(value: object) -> list[str]:
    if not is_object_list(value):
        return []
    return [item for item in value if isinstance(item, str)]


def sanitize_introspection_node(node: dict[str, object]) -> IntrospectionGraphNodeWire:
    """Strip secrets from a manifest node dict, keep observable fields."""
    name_raw = node.get("name")
    type_raw = node.get("type")
    result: IntrospectionGraphNodeWire = {
        "name": name_raw if isinstance(name_raw, str) else "?",
        "type": type_raw if isinstance(type_raw, str) else "?",
    }
    model = node.get("model")
    if isinstance(model, str) and model:
        result["model"] = model
    if node.get("pii_masking"):
        result["pii_masking"] = True
    tool_binding = node.get("tool_binding")
    if isinstance(tool_binding, str):
        result["tool_binding"] = tool_binding
    elif is_object_list(tool_binding):
        binding_list = _string_list(tool_binding)
        if binding_list:
            result["tool_binding"] = binding_list
    tools = node.get("tools")
    if is_object_list(tools):
        tool_names = _string_list(tools)
        if tool_names:
            result["tools"] = tool_names
    return result


def sanitize_introspection_edge(edge: dict[str, object]) -> IntrospectionGraphEdgeJsonWire:
    """Keep edge structure for JSON wire (key ``from``), drop internals."""
    from_value = edge.get("from")
    result: IntrospectionGraphEdgeJsonWire = {
        "from": from_value if isinstance(from_value, str) else "?",
    }
    to_value = edge.get("to")
    if isinstance(to_value, str):
        result["to"] = to_value
    condition_key = edge.get("condition_key")
    if isinstance(condition_key, str):
        result["condition_key"] = condition_key
        condition_map = edge.get("condition_map")
        mapped = _string_dict(condition_map)
        if mapped:
            result["condition_map"] = mapped
    return result


def sanitize_introspection_graph(
    graph_key: str, gdata: dict[str, object]
) -> IntrospectionGraphJsonWire:
    """Sanitize a single graph entry for introspection JSON wire."""
    result: IntrospectionGraphJsonWire = {"name": graph_key}

    template = gdata.get("template")
    if isinstance(template, str) and template:
        result["source"] = template
    else:
        builtin = gdata.get("builtin")
        if isinstance(builtin, str) and builtin:
            result["source"] = f"builtin:{builtin}"
        else:
            result["source"] = "inline"

    nodes = gdata.get("nodes")
    if is_object_list(nodes):
        parsed_nodes = [sanitize_introspection_node(node) for node in nodes if is_object_dict(node)]
        if parsed_nodes:
            result["nodes"] = parsed_nodes

    edges = gdata.get("edges")
    if is_object_list(edges):
        parsed_edges = [sanitize_introspection_edge(edge) for edge in edges if is_object_dict(edge)]
        if parsed_edges:
            result["edges"] = parsed_edges

    return result


def sanitize_introspection_policy(policy: dict[str, object]) -> IntrospectionPolicyWire:
    """Strip secret refs from policy for introspection wire."""
    result: IntrospectionPolicyWire = {}
    models_raw = policy.get("models")
    if is_object_dict(models_raw):
        llm_raw = models_raw.get("llm")
        if is_object_dict(llm_raw):
            default_llm = llm_raw.get("default")
            if isinstance(default_llm, str):
                result["default_llm"] = default_llm
            fallbacks = llm_raw.get("fallback")
            if is_object_list(fallbacks):
                fallback_llms = _string_list(fallbacks)
                if fallback_llms:
                    result["fallback_llms"] = fallback_llms
        embeddings_raw = models_raw.get("embeddings")
        if is_object_dict(embeddings_raw):
            default_embeddings = embeddings_raw.get("default")
            if isinstance(default_embeddings, str):
                result["default_embeddings"] = default_embeddings
    langfuse_raw = policy.get("langfuse")
    if is_object_dict(langfuse_raw):
        tracing_enabled = langfuse_raw.get("tracing_enabled", False)
        if isinstance(tracing_enabled, bool):
            result["langfuse_tracing"] = tracing_enabled
    return result


def sanitize_introspection_services(raw: object) -> dict[str, IntrospectionServiceWire]:
    """Narrow manifest ``services`` map for introspection wire."""
    if not is_object_dict(raw):
        return {}
    out: dict[str, IntrospectionServiceWire] = {}
    for service_name, service_payload in raw.items():
        if not is_object_dict(service_payload):
            continue
        entry: IntrospectionServiceWire = {}
        enabled = service_payload.get("enabled")
        if isinstance(enabled, bool):
            entry["enabled"] = enabled
        out[service_name] = entry
    return out


def build_introspection_tools_from_config(tools_raw: object) -> list[IntrospectionToolWire]:
    """Build tool rows from in-memory manifest ``tools`` config."""
    tools: list[IntrospectionToolWire] = []
    if not is_object_list(tools_raw):
        return tools
    for tool_row in tools_raw:
        if is_object_dict(tool_row):
            name_raw = tool_row.get("name")
            type_raw = tool_row.get("type")
            tools.append(
                {
                    "name": name_raw if isinstance(name_raw, str) else "?",
                    "type": type_raw if isinstance(type_raw, str) else "?",
                }
            )
        elif isinstance(tool_row, str):
            tools.append({"name": tool_row, "type": "federated"})
    return tools


def build_introspection_project_json(
    *,
    project_id: str,
    tools: list[IntrospectionToolWire],
    graphs: dict[str, IntrospectionGraphJsonWire],
    services: dict[str, IntrospectionServiceWire],
    policy: IntrospectionPolicyWire,
) -> IntrospectionProjectJsonWire:
    """Assemble one project record for ``IntrospectRegistrations`` payload."""
    entry: IntrospectionProjectJsonWire = {
        "project_id": project_id,
        "tools": tools,
        "graphs": graphs,
        "services": services,
        "policy": policy,
    }
    return entry


def introspection_registrations_payload(
    projects: list[IntrospectionProjectJsonWire],
) -> dict[str, object]:
    """Wrap projects for ``make_response`` (L3 ``ContextUnit`` open map)."""
    return {"projects": projects}


__all__ = [
    "IntrospectionGraphEdgeJsonWire",
    "IntrospectionGraphEdgeWire",
    "IntrospectionGraphJsonWire",
    "IntrospectionGraphNodeWire",
    "IntrospectionGraphWire",
    "IntrospectionPolicyWire",
    "IntrospectionProjectJsonWire",
    "IntrospectionProjectWire",
    "IntrospectionServiceWire",
    "IntrospectionToolWire",
    "build_introspection_project_json",
    "build_introspection_tools_from_config",
    "introspection_registrations_payload",
    "sanitize_introspection_edge",
    "sanitize_introspection_graph",
    "sanitize_introspection_node",
    "sanitize_introspection_policy",
    "sanitize_introspection_services",
]
