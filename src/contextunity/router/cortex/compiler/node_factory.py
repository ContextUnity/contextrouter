"""Node construction for compiled graphs — builds LangGraph nodes from manifest topology declarations."""

from __future__ import annotations

from typing import Protocol

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict

from contextunity.router.cortex.types import NodeFunc

from .node_config import NodeConfig
from .node_executors.agent import make_agent_node
from .node_executors.llm import make_llm_node
from .node_executors.platform import make_platform_node
from .topology import normalize_tool_ref, parse_binding
from .types import CompilerNodeSpec, ProjectManifest

logger = get_contextunit_logger(__name__)


class _NodeAddingGraph(Protocol):
    """Minimal graph surface needed by the compiler."""

    def add_node(self, node: str, action: NodeFunc) -> object: ...


def _resolve_binding(
    node_type: str, node_spec: CompilerNodeSpec, manifest: ProjectManifest
) -> tuple[str, str]:
    """Parse *tool_binding* on tool-type nodes into a ``(kind, name)`` pair via ``federated_tool_map`` lookup."""
    binding_value = node_spec.get("tool_binding")
    if node_type != "tool" or not isinstance(binding_value, str) or not binding_value:
        return "", ""

    parsed_kind, parsed_name = parse_binding(binding_value)
    if parsed_kind == "platform":
        return "platform", parsed_name

    binding_name = parsed_name or binding_value
    resolved_tools_raw = manifest.get("federated_tool_map", {})
    if is_object_dict(resolved_tools_raw):
        resolved = resolved_tools_raw.get(binding_name)
        if isinstance(resolved, str):
            binding_name = resolved
    return "federated", binding_name


def _platform_required_scopes(tool_name: str) -> list[str]:
    """Delegate to the platform executor module to retrieve required token scopes for *tool_name*."""
    from contextunity.router.cortex.compiler.node_executors.platform import (
        get_platform_required_scopes,
    )

    return get_platform_required_scopes(tool_name)


def _resolve_manifest_tool_name(tool_name: str, manifest: ProjectManifest) -> str:
    """Map manifest tool aliases through ``federated_tool_map`` (same as tool nodes)."""
    federated_map = manifest.get("federated_tool_map")
    if is_object_dict(federated_map):
        mapped = federated_map.get(tool_name)
        if isinstance(mapped, str) and mapped:
            return mapped
    return tool_name


def _agent_execute_tools(node_spec: CompilerNodeSpec, manifest: ProjectManifest) -> list[str]:
    """Collect normalised tool names from the node’s ``tools`` list and resolved ``federated_tool_map``."""
    execute_tools: list[str] = []
    for tool_ref in node_spec.get("tools", []):
        bare_name = normalize_tool_ref(str(tool_ref))
        execute_tools.append(_resolve_manifest_tool_name(bare_name, manifest))
    if node_spec.get("toolkits"):
        from contextunity.router.core.exceptions import RouterGraphBuilderError

        raise RouterGraphBuilderError(
            message="Agent node toolkits must be expanded to explicit tools by the SDK bundle generator"
        )
    return execute_tools


def add_compiled_node(
    graph: _NodeAddingGraph,
    node_spec: CompilerNodeSpec,
    *,
    manifest: ProjectManifest,
    router_defaults: NodeConfig,
    entry_nodes: set[str],
    json_required_nodes: set[str],
) -> None:
    """Resolve config/model/binding, build the correct executor, secure-wrap it, and add to the *graph*."""
    from contextunity.router.core.exceptions import RouterGraphBuilderError
    from contextunity.router.cortex.compiler.config_resolver import (
        resolve_model,
        resolve_model_secret_ref,
        resolve_node_config,
    )
    from contextunity.router.cortex.secure_node import make_secure_node

    name = node_spec.get("name")
    if not name:
        raise RouterGraphBuilderError(
            message=f"Node spec missing required 'name' field: {node_spec!r}"
        )
    node_type = node_spec.get("type", "llm")
    logger.debug("Adding custom node: %s (type: %s)", name, node_type)

    resolved_config = resolve_node_config(node_spec, manifest, router_defaults)
    resolved_model = resolve_model(node_spec, manifest, router_defaults)
    resolved_secret_ref = resolve_model_secret_ref(node_spec, manifest, router_defaults)

    resolved_node_spec: CompilerNodeSpec = {**node_spec}
    resolved_node_spec["config"] = resolved_config

    if name not in entry_nodes and "state_input_key" not in resolved_config:
        resolved_config["state_input_key"] = "final_output"
    if name in json_required_nodes and "output_format" not in resolved_config:
        resolved_config["output_format"] = "json"
    if resolved_config.get("output_format") == "json":
        if (
            resolved_config.get("max_tokens") is None
            and resolved_config.get("max_output_tokens") is None
        ):
            resolved_config["max_tokens"] = 4096
    if resolved_model:
        resolved_node_spec["model"] = resolved_model
    if resolved_secret_ref:
        resolved_node_spec["model_secret_ref"] = resolved_secret_ref

    # Resolve tool bindings for tool-type nodes
    binding_kind = ""
    binding_name = ""
    if node_type == "tool":
        binding_kind, binding_name = _resolve_binding(node_type, resolved_node_spec, manifest)
        if binding_kind and binding_name:
            resolved_node_spec["tool_binding"] = f"{binding_kind}:{binding_name}"
            resolved_node_spec["tool_name"] = binding_name
            resolved_node_spec["tool_kind"] = binding_kind

    is_injected_node = name.startswith("_")
    execute_tools = _agent_execute_tools(node_spec, manifest)
    if (
        not is_injected_node
        and node_type == "tool"
        and binding_kind == "federated"
        and binding_name
        and binding_name not in execute_tools
    ):
        execute_tools.append(binding_name)

    # ── Build executor + secure wrapper ──────────────────────────────

    if node_type == "agent":
        _ = graph.add_node(
            name,
            make_secure_node(
                name,
                make_agent_node(resolved_node_spec, manifest),
                resolved_node_spec,
                execute_tools=execute_tools or None,
            ),
        )
        return

    if node_type == "llm":
        _ = graph.add_node(
            name,
            make_secure_node(
                name,
                make_llm_node(resolved_node_spec, manifest),
                resolved_node_spec,
                execute_tools=execute_tools or None,
            ),
        )
        return

    if node_type == "tool":
        if binding_kind == "platform":
            platform_executor = make_platform_node(resolved_node_spec, manifest)
            if is_injected_node:
                _ = graph.add_node(
                    name,
                    make_secure_node(
                        name,
                        platform_executor,
                        resolved_node_spec,
                        requires_llm=False,
                        pass_through_token=True,
                    ),
                )
                return
            _ = graph.add_node(
                name,
                make_secure_node(
                    name,
                    platform_executor,
                    resolved_node_spec,
                    requires_llm=False,
                    service_scopes=_platform_required_scopes(
                        resolved_node_spec.get("tool_name", "")
                    ),
                ),
            )
            return

        from contextunity.router.cortex.compiler.node_executors.federated import make_federated_node

        _ = graph.add_node(
            name,
            make_secure_node(
                name,
                make_federated_node(resolved_node_spec, manifest),
                resolved_node_spec,
                requires_llm=False,
                execute_tools=execute_tools or None,
            ),
        )
        return

    raise RouterGraphBuilderError(
        message=(
            f"Unsupported node type '{node_type}' on node '{name}'. "
            "Supported types: llm, agent, tool. Platform routing uses tool_binding."
        )
    )


__all__ = ["add_compiled_node"]
