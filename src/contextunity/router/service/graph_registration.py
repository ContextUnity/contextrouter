"""Graph registration helpers for RegisterManifest."""

from __future__ import annotations

from typing import Protocol

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, SecurityError
from contextunity.core.manifest.helpers import parse_tool_ref as _parse_tool_binding
from contextunity.core.manifest.router import RouterEdge, RouterNode
from contextunity.core.types import is_json_dict, is_object_dict

from contextunity.router.core.exceptions import RouterGraphBuilderError
from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    ProjectManifest,
    ServiceDependencyConfig,
    TemplateBuildOverrides,
    TemplateNodeOverride,
    coerce_manifest_int,
    is_template_node_override,
)
from contextunity.router.cortex.types import GraphFactoryProduct
from contextunity.router.service.mixins.execution.types import (
    ProjectConfigMap,
    ProjectGraphMap,
    ProjectToolMap,
    RouterCallbackMap,
)
from contextunity.router.service.payloads import GraphEntry
from contextunity.router.service.registration_projection import coerce_compiler_node_spec

logger = get_contextunit_logger(__name__)


class _CompilableWorkflow(Protocol):
    """Structural type for LangGraph ``StateGraph.compile`` without vendor stubs."""

    def compile(self) -> GraphFactoryProduct:
        """Compile workflow into a registry-ready graph product."""
        ...


def _compile_workflow(workflow: _CompilableWorkflow) -> GraphFactoryProduct:
    """Compile a workflow through a typed structural boundary."""
    return workflow.compile()


def _router_node_to_compiler_spec(node: RouterNode) -> CompilerNodeSpec:
    """Convert a manifest ``RouterNode`` into a compiler node spec."""
    dumped = node.model_dump(exclude_none=True, by_alias=False)
    if not is_json_dict(dumped):
        raise ConfigurationError(message=f"Invalid node spec for '{node.name}'")
    return coerce_compiler_node_spec(dumped)


def _router_edge_to_compiler_spec(edge: RouterEdge) -> CompilerEdgeSpec:
    """Convert a manifest ``RouterEdge`` into a compiler edge spec."""
    spec: CompilerEdgeSpec = {"from_node": edge.from_node}
    if edge.to_node is not None:
        spec["to_node"] = edge.to_node
    if edge.condition_key is not None:
        spec["condition_key"] = edge.condition_key
    if edge.condition_map is not None:
        spec["condition_map"] = dict(edge.condition_map)
    return spec


def _validate_template_federated_tools(
    template_name: str,
    available_tools: set[str],
    resolved_tools: dict[str, str] | None = None,
) -> None:
    """Validate that all federated tools required by a YAML template are registered.

    Args:
        template_name: YAML template identifier to load and inspect.
        available_tools: Tool names currently registered for the project.
        resolved_tools: Optional logical-to-actual name mapping for aliased tools.

    Raises:
        ConfigurationError: If required federated tools are missing.
    """
    from contextunity.router.cortex.compiler.template_loader import load_template

    template = load_template(template_name)
    required: set[str] = set()
    for node in template.nodes:
        if node.type != "tool" or not node.tool_binding:
            continue
        kind, tool_name = _parse_tool_binding(node.tool_binding)
        if kind == "federated" and tool_name:
            required.add(tool_name)

    resolved_tools = resolved_tools or {}
    satisfied = set(available_tools)
    for logical_name, actual_name in resolved_tools.items():
        if actual_name in available_tools:
            satisfied.add(logical_name)

    missing = required - satisfied
    if missing:
        raise ConfigurationError(
            message=(
                f"Template '{template_name}' requires federated tools not registered "
                f"by toolkits or explicit tool bindings: {sorted(missing)}. "
                f"Available: {sorted(satisfied)}."
            ),
        )


def _federated_tool_map(
    config: dict[str, object] | ProjectManifest | None,
) -> dict[str, str] | None:
    """Extract the ``federated_tool_map`` from graph config, if present.

    Args:
        config: Graph configuration dict (may be ``None``).

    Returns:
        Logical-to-actual tool name mapping, or ``None``.
    """
    if config is None:
        return None
    resolved_raw = config.get("federated_tool_map")
    if not is_object_dict(resolved_raw):
        return None
    tool_map: dict[str, str] = {}
    for raw_key, raw_value in resolved_raw.items():
        if not isinstance(raw_value, str):
            return None
        tool_map[raw_key] = raw_value
    return tool_map


def _service_dependency_map(
    services_raw: object,
) -> dict[str, ServiceDependencyConfig] | None:
    """Coerce manifest ``services`` into typed service dependency entries."""
    if not is_object_dict(services_raw):
        return None

    services: dict[str, ServiceDependencyConfig] = {}
    for raw_key, raw_value in services_raw.items():
        if not is_object_dict(raw_value):
            raise ConfigurationError(message=f"Service dependency '{raw_key}' must be an object")
        svc_config = raw_value
        entry: ServiceDependencyConfig = {}
        enabled = svc_config.get("enabled")
        if isinstance(enabled, bool):
            entry["enabled"] = enabled
        url = svc_config.get("url")
        if isinstance(url, str):
            entry["url"] = url
        timeout = svc_config.get("timeout")
        if isinstance(timeout, int):
            entry["timeout"] = timeout
        if not entry:
            raise ConfigurationError(
                message=f"Service dependency '{raw_key}' must declare at least one valid field"
            )
        services[raw_key] = entry
    return services or None


def _project_manifest_config(config: dict[str, object] | None) -> ProjectManifest | None:
    """Coerce a free-form graph config map into the compiler manifest subset."""
    if not config:
        return None

    graph_config = {str(key): value for key, value in config.items() if key != "config"}
    nested_config = config.get("config")
    if is_json_dict(nested_config):
        graph_config.update(dict(nested_config))
    manifest: ProjectManifest = {"config": graph_config}

    for key in ("model", "model_secret_ref", "goal", "persona"):
        value = config.get(key)
        if isinstance(value, str):
            manifest[key] = value

    for key in ("max_retries", "timeout"):
        value = config.get(key)
        coerced = coerce_manifest_int(value)
        if coerced is not None:
            manifest[key] = coerced

    if is_json_dict(nested_config):
        if "max_retries" not in manifest:
            nested_retries = coerce_manifest_int(nested_config.get("max_retries"))
            if nested_retries is not None:
                manifest["max_retries"] = nested_retries
        if "timeout" not in manifest:
            nested_timeout = coerce_manifest_int(nested_config.get("timeout"))
            if nested_timeout is not None:
                manifest["timeout"] = nested_timeout

    services = _service_dependency_map(config.get("services"))
    if services is not None:
        manifest["services"] = services

    tool_map = _federated_tool_map(config)
    if tool_map is not None:
        manifest["federated_tool_map"] = tool_map

    return manifest or None


def _template_build_overrides(graph_config: GraphEntry) -> dict[str, TemplateNodeOverride]:
    """Coerce graph entry overrides to compiler template override contracts."""
    overrides: dict[str, TemplateNodeOverride] = {}
    for node_name, node_overrides in (graph_config.overrides or {}).items():
        if is_template_node_override(node_overrides):
            overrides[node_name] = node_overrides
    return overrides


def _compile_yaml_graph(
    *,
    project_id: str,
    registry_name: str,
    template_name: str,
    template_label: str,
    config: ProjectManifest | None,
    overrides: TemplateBuildOverrides,
) -> str:
    """Build and register a graph from a YAML template definition."""
    try:
        from contextunity.router.cortex.compiler.builder import build_from_template

        workflow = build_from_template(
            template_name=template_name,
            overrides=overrides,
            config=config,
        )
        compiled_graph = _compile_workflow(workflow)
    except (ConfigurationError, RouterGraphBuilderError, SecurityError):
        logger.exception(
            "Failed to compile yaml template '%s' (%s) for project '%s'. Registry left unchanged.",
            template_name,
            template_label,
            project_id,
        )
        raise

    def builder() -> GraphFactoryProduct:
        """Return the pre-compiled graph (closure captures the instance)."""
        return compiled_graph

    from contextunity.router.core.registry import as_runnable_graph_factory, graph_registry

    graph_registry.register(registry_name, as_runnable_graph_factory(builder), overwrite=True)
    logger.info(
        "Registered yaml graph '%s' (template=%s) for project '%s' (atomic)",
        registry_name,
        template_label,
        project_id,
    )
    return registry_name


def _register_yaml_graph(
    project_id: str,
    registry_name: str,
    graph_config: GraphEntry,
    *,
    template_name: str,
    template_label: str,
    available_tools: set[str],
) -> str:
    """Validate federated tools and compile a YAML template graph.

    Args:
        project_id: Owning project identifier.
        registry_name: Namespaced registry key.
        graph_config: Graph entry with template and optional overrides.
        template_name: YAML template name to load.
        template_label: Human-readable label for logging.
        available_tools: Currently registered tool names for the project.

    Returns:
        The registered graph's registry name.
    """
    overrides = _template_build_overrides(graph_config)
    config = _project_manifest_config(graph_config.config)
    _validate_template_federated_tools(
        template_name,
        available_tools,
        _federated_tool_map(config),
    )
    return _compile_yaml_graph(
        project_id=project_id,
        registry_name=registry_name,
        template_name=template_name,
        template_label=template_label,
        config=config,
        overrides=overrides,
    )


def _register_local_graph(project_id: str, registry_name: str, graph_config: GraphEntry) -> str:
    """Compile and register an inline (nodes/edges) graph definition.

    Args:
        project_id: Owning project identifier.
        registry_name: Namespaced registry key.
        graph_config: Graph entry with inline ``nodes`` and ``edges``.

    Returns:
        The registered graph's registry name.
    """
    try:
        from contextunity.router.cortex.compiler.builder import build_local_graph

        compiler_config = _project_manifest_config(graph_config.config) or {}
        nodes = [_router_node_to_compiler_spec(node) for node in (graph_config.nodes or [])]
        edges = [_router_edge_to_compiler_spec(edge) for edge in (graph_config.edges or [])]

        workflow = build_local_graph(nodes, edges, compiler_config)
        compiled_graph = _compile_workflow(workflow)
    except (RouterGraphBuilderError, SecurityError):
        logger.exception(
            "Failed to compile local graph '%s' for project '%s'. Registry left unchanged.",
            registry_name,
            project_id,
        )
        raise

    def builder() -> GraphFactoryProduct:
        """Return the pre-compiled graph (closure captures the instance)."""
        return compiled_graph

    from contextunity.router.core.registry import as_runnable_graph_factory, graph_registry

    graph_registry.register(registry_name, as_runnable_graph_factory(builder), overwrite=True)
    logger.info("Registered local graph '%s' for project '%s' (atomic)", registry_name, project_id)
    return registry_name


def register_graph_for_project(
    project_id: str,
    graph_config: GraphEntry,
    *,
    available_tools: set[str] | None = None,
) -> str:
    """Compile and register a project graph under a namespaced registry key.

    Selects the compilation path based on graph source type: inline
    nodes/edges, YAML template, or builtin dispatcher.

    Args:
        project_id: Owning project identifier.
        graph_config: Graph entry specifying the source and configuration.
        available_tools: Registered tool names for federated validation.

    Returns:
        The namespaced registry key for the compiled graph.

    Raises:
        ConfigurationError: If no valid graph source is defined.
    """
    registry_name = f"project:{project_id}:{graph_config.name}"
    available_tools = available_tools or set()

    if graph_config.nodes is not None or graph_config.edges is not None:
        return _register_local_graph(project_id, registry_name, graph_config)

    if graph_config.template:
        if not graph_config.template.startswith("yaml:"):
            raise ConfigurationError(
                message="Graph template source must use 'yaml:<template_name>'"
            )
        yaml_template_name = graph_config.template[5:]
        return _register_yaml_graph(
            project_id,
            registry_name,
            graph_config,
            template_name=yaml_template_name,
            template_label=yaml_template_name,
            available_tools=available_tools,
        )

    if graph_config.builtin == "dispatcher":
        from contextunity.router.core.registry import graph_registry
        from contextunity.router.cortex.dispatcher_agent import compile_dispatcher_graph

        def _dispatcher_graph_builder() -> GraphFactoryProduct:
            return compile_dispatcher_graph()

        from contextunity.router.core.registry import as_runnable_graph_factory

        graph_registry.register(
            registry_name,
            as_runnable_graph_factory(_dispatcher_graph_builder),
            overwrite=True,
        )
        logger.info(
            "Registered graph '%s' (builtin=%s) for project '%s'",
            registry_name,
            graph_config.builtin,
            project_id,
        )
        return registry_name

    raise ConfigurationError(
        message=(
            "Graph config must define exactly one graph source: "
            "inline nodes/edges, template='yaml:<template_name>', or builtin='dispatcher'."
        ),
    )


def deregister_project_graphs(
    *,
    project_tools: ProjectToolMap,
    project_graphs: ProjectGraphMap,
    project_router_callbacks: RouterCallbackMap,
    project_configs: ProjectConfigMap | None = None,
    project_id: str,
) -> list[str]:
    """Remove all tools and graph registry entries for a project.

    Cleans up federated tools, graph registry entries, and
    router callback maps associated with the project.

    Args:
        project_tools: Per-project registered tool names.
        project_graphs: Per-project graph registry map.
        project_router_callbacks: Per-project router callback map.
        project_id: Project to deregister.

    Returns:
        List of deregistered tool and graph names.
    """
    from contextunity.router.modules.tools import deregister_tool

    deregistered: list[str] = []

    tool_names = project_tools.pop(project_id, [])
    for name in tool_names:
        if deregister_tool(name, project_id=project_id):
            deregistered.append(name)

    graph_entry = project_graphs.pop(project_id, None)
    if graph_entry:
        from contextunity.router.core.registry import graph_registry

        unique_graphs = {val for key, val in graph_entry.items() if key != "default"}
        for item in unique_graphs:
            _ = graph_registry.unregister(item)
            logger.info("Deregistered graph '%s' (project '%s')", item, project_id)
            deregistered.append(f"graph:{item}")

    _ = project_router_callbacks.pop(project_id, None)
    if project_configs is not None:
        _ = project_configs.pop(project_id, None)

    return deregistered
