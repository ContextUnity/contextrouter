"""L4 projection from RouterRegistrationBundle wire dict to RegisteredProjectConfig."""

from __future__ import annotations

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.manifest.tenants import parse_allowed_tenants_field
from contextunity.core.sdk.payload import get_dict_list, get_json_dict
from contextunity.core.types import (
    ContextUnitPayload,
    JsonDict,
    is_json_dict,
    is_object_dict,
    is_object_list,
)

from contextunity.router.cortex.compiler.types import CompilerNodeSpec, NodeMeta
from contextunity.router.cortex.types import (
    RegisteredGraphMap,
    RegisteredProjectConfig,
    RegisteredToolEntry,
)

logger = get_contextunit_logger(__name__)


def coerce_compiler_node_spec(raw: JsonDict) -> CompilerNodeSpec:
    """Narrow a graph entry node dict into ``CompilerNodeSpec`` (bundle wire → L4 store)."""
    spec: CompilerNodeSpec = {}
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigurationError(message=f"Node entry missing non-empty name: {raw!r}")
    spec["name"] = name
    node_type = raw.get("type")
    if node_type in ("llm", "embeddings", "agent", "tool"):
        spec["type"] = node_type
    elif node_type is not None:
        raise ConfigurationError(message=f"Invalid node type for '{name}': {node_type!r}")
    elif any(key in raw for key in ("tool_binding", "tools", "toolkits")):
        raise ConfigurationError(
            message=f"Node '{name}' declares tool bindings without an explicit valid type",
        )
    mode = raw.get("mode")
    if isinstance(mode, str):
        spec["mode"] = mode
    model = raw.get("model")
    if isinstance(model, str):
        spec["model"] = model
    prompt_ref = raw.get("prompt_ref")
    if isinstance(prompt_ref, str):
        spec["prompt_ref"] = prompt_ref
    prompt_variants_ref = raw.get("prompt_variants_ref")
    if isinstance(prompt_variants_ref, str):
        spec["prompt_variants_ref"] = prompt_variants_ref
    prompt_version = raw.get("prompt_version")
    if isinstance(prompt_version, str):
        spec["prompt_version"] = prompt_version
    prompt_signature = raw.get("prompt_signature")
    if isinstance(prompt_signature, str):
        spec["prompt_signature"] = prompt_signature
    prompt_variants_versions = raw.get("prompt_variants_versions")
    if is_json_dict(prompt_variants_versions):
        spec["prompt_variants_versions"] = {
            str(key): value
            for key, value in prompt_variants_versions.items()
            if isinstance(value, str)
        }
    description = raw.get("description")
    if isinstance(description, str):
        spec["description"] = description
    model_secret_ref = raw.get("model_secret_ref")
    if isinstance(model_secret_ref, str):
        spec["model_secret_ref"] = model_secret_ref
    goal = raw.get("goal")
    if isinstance(goal, str):
        spec["goal"] = goal
    persona = raw.get("persona")
    if isinstance(persona, str):
        spec["persona"] = persona
    tool_name = raw.get("tool_name")
    if isinstance(tool_name, str):
        spec["tool_name"] = tool_name
    tool_kind = raw.get("tool_kind")
    if isinstance(tool_kind, str):
        spec["tool_kind"] = tool_kind
    tool_binding = raw.get("tool_binding")
    if isinstance(tool_binding, str):
        spec["tool_binding"] = tool_binding
    if raw.get("pii_masking") is True:
        spec["pii_masking"] = True
    config = raw.get("config")
    if is_json_dict(config):
        spec["config"] = {str(k): v for k, v in config.items()}
    meta = raw.get("meta")
    if is_object_dict(meta):
        node_meta: NodeMeta = {}
        for meta_key in ("tool_kind", "source", "toolkit"):
            meta_val = meta.get(meta_key)
            if isinstance(meta_val, str):
                node_meta[meta_key] = meta_val
        if node_meta:
            spec["meta"] = node_meta
    tools = raw.get("tools")
    if isinstance(tools, list):
        tool_names = [item for item in tools if isinstance(item, str)]
        if len(tool_names) != len(tools):
            raise ConfigurationError(message=f"Node '{name}' has non-string tools entries")
        spec["tools"] = tool_names
    toolkits = raw.get("toolkits")
    if isinstance(toolkits, list):
        toolkit_names = [item for item in toolkits if isinstance(item, str)]
        if len(toolkit_names) != len(toolkits):
            raise ConfigurationError(message=f"Node '{name}' has non-string toolkits entries")
        spec["toolkits"] = toolkit_names
    graph_key = raw.get("graph_key")
    if isinstance(graph_key, str) and graph_key:
        spec["graph_key"] = graph_key
    allowed_tenants = raw.get("allowed_tenants")
    if isinstance(allowed_tenants, list):
        tenants = [item for item in allowed_tenants if isinstance(item, str) and item]
        if tenants:
            spec["allowed_tenants"] = tenants
    return spec


def _tool_dict_to_registered(raw: JsonDict) -> RegisteredToolEntry | None:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return None
    entry: RegisteredToolEntry = {"name": name}
    tool_type = raw.get("type")
    if isinstance(tool_type, str):
        entry["type"] = tool_type
    description = raw.get("description")
    if isinstance(description, str):
        entry["description"] = description
    config = raw.get("config")
    if is_json_dict(config):
        entry["config"] = config
    return entry


def tools_from_bundle_tools(raw_tools: list[dict[str, object]]) -> list[RegisteredToolEntry]:
    """Narrow bundle ``tools`` list into ``RegisteredToolEntry`` records."""
    result: list[RegisteredToolEntry] = []
    for index, raw in enumerate(raw_tools):
        if not is_json_dict(raw):
            raise ConfigurationError(message=f"Tool entry at index {index} must be JSON object")
        entry = _tool_dict_to_registered(raw)
        if entry is None:
            raise ConfigurationError(message=f"Tool entry at index {index} missing non-empty name")
        result.append(entry)
    return result


def _graph_map_from_json(graph_map: JsonDict) -> RegisteredGraphMap:
    result: RegisteredGraphMap = {}
    for key, value in graph_map.items():
        if not is_json_dict(value):
            raise ConfigurationError(message=f"Graph entry '{key}' must be JSON object")
        result[key] = value
    return result


def nodes_from_graph_map(graph_map: JsonDict) -> list[CompilerNodeSpec]:
    """Merge denormalized ``nodes[]`` from all graph entries in the bundle."""
    all_nodes: list[CompilerNodeSpec] = []
    for graph_key, raw_entry in graph_map.items():
        if not is_json_dict(raw_entry):
            raise ConfigurationError(message=f"Graph entry '{graph_key}' must be JSON object")
        entry_nodes = raw_entry.get("nodes")
        if not isinstance(entry_nodes, list):
            continue
        for node in entry_nodes:
            if not is_json_dict(node):
                raise ConfigurationError(
                    message=f"Graph entry '{graph_key}' contains a non-object node"
                )
            node_with_key = dict(node)
            if "graph_key" not in node_with_key:
                node_with_key["graph_key"] = graph_key
            all_nodes.append(coerce_compiler_node_spec(node_with_key))
    return all_nodes


def registered_project_config_from_bundle(
    bundle: ContextUnitPayload,
    graph_map: JsonDict,
) -> RegisteredProjectConfig:
    """Project ``RegisterManifest`` bundle fields into the runtime store shape."""
    config: RegisteredProjectConfig = {}
    project_id_raw = bundle.get("project_id")
    if isinstance(project_id_raw, str) and project_id_raw:
        config["project_id"] = project_id_raw
    allowed = parse_allowed_tenants_field(bundle.get("allowed_tenants"))
    if allowed:
        config["allowed_tenants"] = allowed
    config["policy"] = get_json_dict(bundle, "policy")
    config["tools"] = tools_from_bundle_tools(get_dict_list(bundle, "tools"))
    config["services"] = get_json_dict(bundle, "services")
    config["graph"] = _graph_map_from_json(graph_map)
    all_nodes = nodes_from_graph_map(graph_map)
    if all_nodes:
        config["nodes"] = all_nodes
    return config


def registered_project_config_from_persisted(
    payload: dict[str, object],
    graph_map: JsonDict,
) -> RegisteredProjectConfig:
    """Rebuild ``RegisteredProjectConfig`` from a Redis-persisted bundle payload."""
    policy_raw = payload.get("policy")
    services_raw = payload.get("services")
    tools_raw = payload.get("tools")

    tools: list[RegisteredToolEntry] = []
    if is_object_list(tools_raw):
        for index, item_obj in enumerate(tools_raw):
            if is_json_dict(item_obj):
                entry = _tool_dict_to_registered(item_obj)
                if entry is not None:
                    tools.append(entry)
                else:
                    logger.warning(
                        "Persisted registration skipped malformed tool row at index %d",
                        index,
                    )
            else:
                logger.warning(
                    "Persisted registration skipped non-object tool row at index %d",
                    index,
                )

    config: RegisteredProjectConfig = {}
    project_id_raw = payload.get("project_id")
    if isinstance(project_id_raw, str) and project_id_raw:
        config["project_id"] = project_id_raw
    allowed = parse_allowed_tenants_field(payload.get("allowed_tenants"))
    if allowed:
        config["allowed_tenants"] = allowed
    config["policy"] = policy_raw if is_json_dict(policy_raw) else {}
    config["tools"] = tools
    config["services"] = services_raw if is_json_dict(services_raw) else {}
    config["graph"] = _graph_map_from_json(graph_map)
    all_nodes = nodes_from_graph_map(graph_map)
    if all_nodes:
        config["nodes"] = all_nodes
    return config


__all__ = [
    "coerce_compiler_node_spec",
    "nodes_from_graph_map",
    "registered_project_config_from_bundle",
    "registered_project_config_from_persisted",
    "tools_from_bundle_tools",
]
