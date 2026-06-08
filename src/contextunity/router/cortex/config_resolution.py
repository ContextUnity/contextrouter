"""Node-level config lookup from registered project configs — linear scan by node name."""

from __future__ import annotations

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_json_dict, is_object_dict
from typing_extensions import TypedDict

from .compiler.types import CompilerNodeSpec
from .types import GraphState, RegisteredProjectConfig, is_registered_project_config

logger = get_contextunit_logger(__name__)


class GraphRuntimeConfig(TypedDict, total=False):
    """Merged graph-level runtime settings from registered graph entries."""

    goal: str
    persona: str
    model_key: str
    node_tool_bindings: dict[str, object]


def metadata_project_config(state: GraphState) -> RegisteredProjectConfig:
    """Return ``state.metadata.project_config`` when execution injected L4 config."""
    metadata_raw = state.get("metadata")
    if not is_object_dict(metadata_raw):
        return {}
    project_config_raw = metadata_raw.get("project_config")
    if is_registered_project_config(project_config_raw):
        return project_config_raw
    return {}


def metadata_project_id(state: GraphState) -> str:
    """Return the trusted project id injected in runtime project config."""
    project_id = metadata_project_config(state).get("project_id")
    return project_id if isinstance(project_id, str) else ""


def get_node_config(project_config: RegisteredProjectConfig, node_name: str) -> CompilerNodeSpec:
    """Scan denormalized ``project_config["nodes"]`` for *node_name*."""
    nodes = project_config.get("nodes", [])
    for node in nodes:
        if node.get("name") == node_name:
            return node
    return {}


def get_node_attr(
    project_config: RegisteredProjectConfig,
    node_name: str,
    attr: str,
    default: object = None,
) -> object:
    """Return ``get_node_config(…)[attr]`` with a *default* fallback."""
    return get_node_config(project_config, node_name).get(attr, default)


def get_node_manifest_config(state: GraphState, node_name: str) -> CompilerNodeSpec:
    """Extract node config from ``state["metadata"]["project_config"]`` — convenience wrapper for node executors."""
    result = get_node_config(metadata_project_config(state), node_name)
    if not result:
        logger.warning("Node '%s' not found in manifest configuration. Using defaults.", node_name)
    return result


def get_graph_runtime_config(project_config: RegisteredProjectConfig) -> GraphRuntimeConfig:
    """Merge graph-level runtime config blobs from all registered graph entries.

    ``ArtifactGenerator`` stores goal/persona/model_key/node_tool_bindings under
    ``graph[graph_key]["config"]``. Runtime ``RegisteredProjectConfig`` does not
    hoist those keys to the top level.
    """
    graph_map = project_config.get("graph")
    if not is_object_dict(graph_map):
        return {}

    merged: GraphRuntimeConfig = {}
    for entry in graph_map.values():
        if not is_json_dict(entry):
            continue
        cfg = entry.get("config")
        if not is_object_dict(cfg):
            continue
        for key, value in cfg.items():
            if key == "node_tool_bindings" and is_object_dict(value):
                existing = merged.get("node_tool_bindings")
                if is_object_dict(existing):
                    merged["node_tool_bindings"] = {**existing, **value}
                else:
                    merged["node_tool_bindings"] = dict(value)
            elif key not in merged:
                merged[key] = value
    return merged


def make_shield_path(node_name: str) -> str:
    """Return the Shield secret subpath for a node's LLM API key (e.g. ``"node_name/model_secret_ref"``)."""
    return f"{node_name}/model_secret_ref"
