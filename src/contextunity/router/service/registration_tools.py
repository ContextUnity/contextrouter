"""Tool registration helpers shared by RegisterManifest and Redis restore."""

from __future__ import annotations

from contextunity.core.exceptions import ConfigurationError
from contextunity.core.types import (
    ContextUnitPayload,
    WireValue,
    is_json_dict,
    is_object_dict,
    is_object_list,
)
from langchain_core.tools import BaseTool

from contextunity.router.service.payloads import ToolConfig
from contextunity.router.service.tool_factory import create_tool_from_config

_FEDERATED_TOOL_TYPES = frozenset({"sql", "bidi", "commerce"})


def _strict_tool_dicts(bundle: ContextUnitPayload) -> list[dict[str, object]]:
    """Return bundle tool entries, failing closed on malformed list members."""
    collected: list[dict[str, object]] = []

    graph_map = bundle.get("graph")
    if is_json_dict(graph_map):
        for graph_key, raw_entry in graph_map.items():
            if not is_object_dict(raw_entry):
                continue
            config = raw_entry.get("config")
            if not is_object_dict(config):
                continue
            federated_tools = config.get("federated_tools")
            if not is_object_list(federated_tools):
                continue
            for index, raw_tool in enumerate(federated_tools):
                if not is_object_dict(raw_tool):
                    raise ConfigurationError(
                        message=(f"Graph '{graph_key}' federated_tools[{index}] must be an object")
                    )
                tool_dict = dict(raw_tool)
                tool_config_raw: object = tool_dict.get("config")
                if tool_config_raw is None:
                    tool_config_raw = {}
                if not is_object_dict(tool_config_raw):
                    raise ConfigurationError(
                        message=f"Graph '{graph_key}' tool config must be an object"
                    )
                merged_config = dict(tool_config_raw)
                _ = merged_config.setdefault("graph_key", graph_key)
                tool_dict["config"] = merged_config
                collected.append(tool_dict)

    raw_tools: object = bundle.get("tools", [])
    if raw_tools is None:
        raw_tools = []
    if not is_object_list(raw_tools):
        raise ConfigurationError(message="RegisterManifest bundle 'tools' must be a list")

    seen_names: set[str] = set()
    for raw_tool in collected:
        name_raw = raw_tool.get("name")
        if isinstance(name_raw, str) and name_raw:
            seen_names.add(name_raw)

    for index, raw_tool in enumerate(raw_tools):
        if not is_object_dict(raw_tool):
            raise ConfigurationError(
                message=f"RegisterManifest bundle tool at index {index} must be an object"
            )
        tool_dict = dict(raw_tool)
        name_raw = tool_dict.get("name")
        if isinstance(name_raw, str) and name_raw and name_raw not in seen_names:
            collected.append(tool_dict)
            seen_names.add(name_raw)

    return collected


def _canonical_tool_config(raw_tool: dict[str, object], *, project_id: str) -> ToolConfig:
    """Validate one tool entry and inject canonical stream routing project_id."""
    raw_type = raw_tool.get("type")
    if isinstance(raw_type, str) and raw_type in _FEDERATED_TOOL_TYPES:
        raw_config: object = raw_tool.get("config", {})
        if raw_config is None:
            raw_config = {}
        if not is_object_dict(raw_config):
            raise ConfigurationError(
                message=f"Tool '{raw_tool.get('name', '?')}' config must be an object"
            )
        configured_project = raw_config.get("project_id")
        if configured_project is not None and configured_project != project_id:
            raise ConfigurationError(
                message=(
                    f"Tool '{raw_tool.get('name', '?')}' project_id "
                    f"{configured_project!r} does not match bundle project_id {project_id!r}"
                )
            )
        config: dict[str, WireValue] = dict(raw_config)
        config["project_id"] = project_id
        raw_tool = {**raw_tool, "config": config}

    return ToolConfig.model_validate(raw_tool)


def create_tools_from_bundle(bundle: ContextUnitPayload, *, project_id: str) -> list[BaseTool]:
    """Build all tools declared by a registration bundle without mutating global registries."""
    tools: list[BaseTool] = []
    for raw_tool in _strict_tool_dicts(bundle):
        tool_def = _canonical_tool_config(raw_tool, project_id=project_id)
        tools.extend(
            create_tool_from_config(
                name=tool_def.name,
                tool_type=tool_def.type,
                description=tool_def.description,
                config=tool_def.config,
            )
        )
    return tools


__all__ = ["create_tools_from_bundle"]
