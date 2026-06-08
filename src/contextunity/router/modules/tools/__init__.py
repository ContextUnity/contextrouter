"""Tool discovery and registration for dispatcher agent."""

from __future__ import annotations

import asyncio
import inspect

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonDict, is_json_dict, is_object_sequence
from langchain_core.tools import BaseTool

from contextunity.router.modules.tools.secure import SecureTool

logger = get_contextunit_logger(__name__)

# Global tool registry — ALL values are SecureTool instances
_tool_registry: dict[str, SecureTool] = {}
# Per-project tool namespace so two projects can register distinct tools
# under the same name (e.g. ``medical_sql`` for two different SQL backends)
# without one silently overwriting the other. Keys are
# ``(project_id, tool_name)``; values are the bound ``SecureTool`` instances.
# Project-aware lookups (``get_tool_for_project``) consult this map first.
_tool_registry_by_project: dict[tuple[str, str], SecureTool] = {}


def _index_tool_in_brain(tool_instance: BaseTool) -> None:
    """Index tool in Brain Procedural Memory for semantic search.

    This is called automatically when a tool is registered.
    The indexing happens asynchronously and does not block tool registration.

    Args:
        tool_instance: Tool instance to index
    """
    # Brain indexing disabled by default — current implementation triggers
    # full ingestion pipeline (taxonomy, NER, graph) for a placeholder result.
    # Enable via CU_BRAIN_INDEX_TOOLS=true when PromptGenerator is ready.
    from contextunity.router.core import get_core_config

    if not get_core_config().router.brain_index_tools:
        logger.debug("Brain tool indexing disabled, skipping %s", tool_instance.name)
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop (e.g., during tests) — skip async indexing
        logger.debug("No event loop, skipping Brain indexing for %s", tool_instance.name)
        return

    # Run async indexing in background (coroutine created inside create_task)
    _ = loop.create_task(_async_index_tool(tool_instance))


async def _async_index_tool(tool_instance: BaseTool) -> None:
    """Asynchronously index tool in Brain Procedural Memory.

    Args:
        tool_instance: Tool instance to index
    """
    try:
        import time

        from contextunity.core import BrainClient

        from contextunity.router.core import get_core_config
        from contextunity.router.core.brain_token import get_brain_service_token

        # Extract tool information
        tool_name = tool_instance.name
        tool_description = tool_instance.description or ""

        # Get tool schema/parameters if available
        tool_schema: JsonDict = {}
        schema_obj: object | None = getattr(tool_instance, "args_schema", None)
        if schema_obj is not None:
            try:
                raw_schema: object | None = None
                if isinstance(schema_obj, type) and hasattr(schema_obj, "model_fields"):
                    model_schema_getter: object = getattr(schema_obj, "model_json_schema", None)
                    if callable(model_schema_getter):
                        raw_schema = model_schema_getter()
                else:
                    json_schema_getter: object = getattr(schema_obj, "model_json_schema", None)
                    if callable(json_schema_getter):
                        raw_schema = json_schema_getter()
                    else:
                        legacy_getter: object = getattr(schema_obj, "schema", None)
                        if callable(legacy_getter):
                            raw_schema = legacy_getter()
                if raw_schema is not None and is_json_dict(raw_schema):
                    tool_schema = raw_schema
            except Exception:
                pass

        # Build content for indexing
        content_parts = [
            f"Tool: {tool_name}",
            f"Description: {tool_description}",
        ]

        if tool_schema:
            properties_raw = tool_schema.get("properties")
            if is_json_dict(properties_raw) and properties_raw:
                content_parts.append("Parameters:")
                for param_name, param_info in properties_raw.items():
                    if not is_json_dict(param_info):
                        continue
                    param_type_raw = param_info.get("type", "unknown")
                    param_type = (
                        str(param_type_raw)
                        if isinstance(param_type_raw, (str, int, float, bool))
                        else "unknown"
                    )
                    param_desc_raw = param_info.get("description", "")
                    param_desc = str(param_desc_raw) if param_desc_raw is not None else ""
                    content_parts.append(f"  - {param_name} ({param_type}): {param_desc}")

        content = "\n".join(content_parts)

        # Create metadata
        metadata: JsonDict = {
            "tool_name": tool_name,
            "tool_type": "procedural",
            "source_type": "procedural",
            "indexed_at": str(time.time()),
        }

        if tool_schema and is_json_dict(tool_schema):
            metadata["tool_schema"] = tool_schema

        # Index in Brain Procedural Memory — use tool tenant binding when present.
        bound_allowed_raw: object = getattr(tool_instance, "bound_allowed_tenants", ())
        bound_tenants_list: list[str] = []
        if is_object_sequence(bound_allowed_raw):
            for tenant_item in bound_allowed_raw:
                if isinstance(tenant_item, str) and tenant_item:
                    bound_tenants_list.append(tenant_item)
        bound_tenants = tuple(bound_tenants_list)
        bound_tenant = getattr(tool_instance, "bound_tenant", None)
        if bound_tenants:
            token_tenants: tuple[str, ...] = bound_tenants
            brain_tenant: str = bound_tenants[0]
        elif isinstance(bound_tenant, str) and bound_tenant:
            token_tenants = (bound_tenant,)
            brain_tenant = bound_tenant
        else:
            token_tenants = ("__global__",)
            brain_tenant = "__global__"

        brain_endpoint = get_core_config().brain_url
        brain = BrainClient(
            host=brain_endpoint,
            token=get_brain_service_token(allowed_tenants=token_tenants),
        )
        item_id = await brain.upsert(
            tenant_id=brain_tenant,
            content=content,
            source_type="procedural",
            metadata=metadata,
        )
        logger.debug("Indexed tool %s in Brain Procedural Memory (id: %s)", tool_name, item_id)

    except Exception as e:
        logger.warning("Failed to index tool %s in Brain: %s", tool_instance.name, e, exc_info=True)


def register_tool(
    tool_instance: BaseTool,
    *,
    permission: str = "",
    scope: str = "read",
    tenant: str = "",
    allowed_tenants: tuple[str, ...] = (),
    project_id: str = "",
) -> None:
    """Register a tool instance in the global or per-project registry.

    If ``tool_instance`` is not already a :class:`SecureTool`, it is
    automatically wrapped with permission enforcement.

    When ``project_id`` is supplied, the tool is indexed only in
    ``_tool_registry_by_project`` under ``(project_id, name)``. This allows
    two projects to register distinct tools under the same global name
    without exposing either implementation to global callers.

    Args:
        tool_instance: Tool instance to register.
        permission: Required permission string (default: ``tool:{name}``).
        scope: Required scope — ``read``, ``write``, or ``admin``.
        tenant: Legacy single-tenant binding.
        allowed_tenants: Full project tenant scope for multi-tenant projects.
        project_id: Optional project namespace. When set, the tool is
            tracked only in the per-project registry.
    """
    # ── Auto-wrap raw BaseTool → SecureTool ──
    if not isinstance(tool_instance, SecureTool):
        tool_instance = SecureTool.wrap(
            tool_instance,
            permission=permission,
            scope=scope,
            tenant=tenant,
            allowed_tenants=allowed_tenants,
        )
        logger.debug(
            "Auto-wrapped tool '%s' → SecureTool (permission=%s, allowed_tenants=%s)",
            tool_instance.name,
            tool_instance.required_permission,
            tool_instance.bound_allowed_tenants or tool_instance.bound_tenant or "(global)",
        )
    else:
        if permission and not tool_instance.required_permission:
            tool_instance.required_permission = permission
        if allowed_tenants and not tool_instance.bound_allowed_tenants:
            tool_instance.bound_allowed_tenants = allowed_tenants
        elif tenant and not tool_instance.bound_tenant:
            tool_instance.bound_tenant = tenant

    name = tool_instance.name
    if project_id:
        _tool_registry_by_project[(project_id, name)] = tool_instance
        logger.debug("Registered tool '%s' under project '%s'", name, project_id)
        _index_tool_in_brain(tool_instance)
        return

    if name in _tool_registry:
        existing = _tool_registry[name]
        existing_scope = existing.bound_allowed_tenants or (
            (existing.bound_tenant,) if existing.bound_tenant else ()
        )
        new_scope = tool_instance.bound_allowed_tenants or (
            (tool_instance.bound_tenant,) if tool_instance.bound_tenant else ()
        )

        if existing_scope != new_scope:
            from contextunity.core.exceptions import SecurityError

            raise SecurityError(
                (
                    f"Tool name collision: '{name}' is already registered "
                    f"for tenants {list(existing_scope) or ['global']}. Cannot overwrite "
                    f"from tenants {list(new_scope) or ['global']}."
                )
            )
        logger.debug("Tool '%s' already registered by same tenant, overwriting", name)
    _tool_registry[name] = tool_instance
    logger.debug("Registered tool: %s", name)

    # Index tool in Brain Procedural Memory (async, non-blocking)
    _index_tool_in_brain(tool_instance)


def discover_all_tools() -> list[BaseTool]:
    """Discover all available tools from various sources.

    All returned tools are guaranteed to be :class:`SecureTool` instances
    with mandatory permission enforcement.  Raw ``BaseTool`` instances
    discovered from modules are auto-wrapped.

    Sources:
    1. Registered tools in _tool_registry
    2. Tools from connector modules
    3. Tools from other registered modules

    Returns:
        List of SecureTool instances
    """
    tools: list[BaseTool] = []

    # NOTE: _tool_registry is populated by module imports below
    # (via register_tool()). We snapshot it at the END of this function.

    # Discover tools from connector modules
    try:
        from contextunity.router.modules.connectors import api, file, rss, web

        # Check each connector module for tools
        for module in [api, file, rss, web]:
            module_attrs: dict[str, object] = dict(vars(module))
            for name, obj in module_attrs.items():
                if inspect.isclass(obj) and issubclass(obj, BaseTool):
                    try:
                        instance = obj()
                        tools.append(instance)
                    except Exception as e:
                        logger.debug("Could not instantiate tool %s: %s", name, e)
    except ImportError as e:
        logger.debug("Could not import connector modules: %s", e)

    # Discover sub-agent tools
    try:
        from contextunity.router.modules.tools import subagent_tools

        # Tools are auto-registered when module is imported
        # Just ensure the module is loaded
        if hasattr(subagent_tools, "spawn_subagent"):
            logger.debug("Sub-agent tools module loaded")
    except ImportError:
        pass

    # Discover Redis memory tools
    try:
        from contextunity.router.modules.tools.redis_memory import (
            cache_query_result,
            clear_memory,
            get_cached_query,
            get_session_data,
            retrieve_memory,
            store_memory,
        )

        tools.extend(
            [
                store_memory,
                retrieve_memory,
                cache_query_result,
                get_cached_query,
                get_session_data,
                clear_memory,
            ]
        )
    except ImportError as e:
        logger.debug("Could not import Redis memory tools: %s", e)

    # Discover Brain memory tools (persistent episodic + entity memory)
    try:
        from contextunity.router.modules.tools.brain_memory_tools import (
            learn_user_fact,
            recall_episodes,
            recall_user_facts,
            remember_episode,
        )

        tools.extend(
            [
                remember_episode,
                recall_episodes,
                learn_user_fact,
                recall_user_facts,
            ]
        )
    except ImportError as e:
        logger.debug("Could not import Brain memory tools: %s", e)

    # Discover SQL tools (if configured)
    try:
        from contextunity.router.modules.tools import sql as _sql_tools  # noqa: F401

        del _sql_tools
        # SQL tools require project-specific config; projects register via register_tool()
    except ImportError as e:
        logger.debug("Could not import SQL tools: %s", e)

    # Discover contextunity.shield security tools (optional — requires contextunity.shield package)
    try:
        from contextunity.router.modules.tools import security_tools as _security_tools

        del _security_tools
        logger.debug("contextunity.shield security tools loaded")
    except ImportError:
        logger.debug("contextunity.shield not installed — security tools unavailable")

    # Discover GCS storage tools (optional — requires google-cloud-storage)
    try:
        from contextunity.router.modules.tools import gcs_tools as _gcs_tools

        del _gcs_tools
        logger.debug("GCS storage tools loaded")
    except ImportError:
        logger.debug("GCS storage tools unavailable (google-cloud-storage not installed)")

    # Discover Brain trace tools (execution trace logging)
    try:
        from contextunity.router.modules.tools import brain_trace_tools as _brain_trace_tools

        del _brain_trace_tools
        logger.debug("Brain trace tools loaded")
    except ImportError as e:
        logger.debug("Could not import Brain trace tools: %s", e)

    # Add all tools that were registered via register_tool()
    # (by module imports above: security_tools, etc.)
    tools.extend(_tool_registry.values())

    # Deduplicate by tool name (some tools may appear in both registry and explicit imports)
    seen: set[str] = set()
    unique_tools: list[BaseTool] = []
    for t in tools:
        if t.name not in seen:
            seen.add(t.name)
            unique_tools.append(t)
    tools = unique_tools

    # ── Ensure ALL tools are SecureTool ──
    secure_tools: list[SecureTool] = []
    for t in tools:
        if isinstance(t, SecureTool):
            secure_tools.append(t)
        else:
            wrapped = SecureTool.wrap(t)
            logger.debug(
                "Tool '%s' discovered as raw BaseTool — auto-wrapped to SecureTool",
                t.name,
            )
            secure_tools.append(wrapped)

    logger.info("Discovered %d tools total (all SecureTool)", len(secure_tools))
    registered: list[BaseTool] = []
    for tool in secure_tools:
        registered.append(tool)
    return registered


def discover_tools_for_project(project_id: str) -> list[BaseTool]:
    """Discover shared tools plus tools registered for one project."""
    by_name = {tool.name: tool for tool in discover_all_tools()}
    if project_id:
        for (registered_project_id, name), tool in _tool_registry_by_project.items():
            if registered_project_id == project_id:
                by_name[name] = tool
    return list(by_name.values())


def get_tool(name: str) -> SecureTool | None:
    """Get a tool by name from the global registry.

    Args:
        name: Tool name

    Returns:
        SecureTool instance or None if not found
    """
    return _tool_registry.get(name)


def get_tool_for_project(project_id: str, name: str) -> SecureTool | None:
    """Get a tool by (project_id, name).

    Use this in tenant-scoped execution paths so two projects can hold
    distinct implementations of the same tool name without ``get_tool()``
    returning the wrong one. Falls back to the global registry when the
    per-project slot is empty — this preserves the built-in tool
    discovery flow for shared tools (e.g. ``learn_user_fact``) that
    are not project-namespaced.
    """
    slot = _tool_registry_by_project.get((project_id, name))
    if slot is not None:
        return slot
    return _tool_registry.get(name)


def deregister_tool(name: str, *, project_id: str = "") -> bool:
    """Remove a tool from the registry by name.

    This is the correct public API for deregistration — callers must
    NOT access ``_tool_registry`` directly.

    Args:
        name: Tool name.
        project_id: When provided, only the per-project slot
            ``(project_id, name)`` is removed. When empty, the global
            entry is removed (legacy behavior).

    Returns:
        True if the tool was found and removed, False if not found.
    """
    if project_id:
        key = (project_id, name)
        if key in _tool_registry_by_project:
            del _tool_registry_by_project[key]
            logger.info("Deregistered tool '%s' from project '%s'", name, project_id)
            return True
        logger.debug("Deregister: tool '%s' not found under project '%s'", name, project_id)
        return False
    if name in _tool_registry:
        del _tool_registry[name]
        logger.info("Deregistered tool: %s", name)
        return True
    logger.debug("Deregister: tool '%s' not found in registry", name)
    return False


def list_project_tools(project_id: str) -> list[str]:
    """List tool names registered for *project_id* via ``register_tool(project_id=...)``.

    Returns:
        List of tool names; empty if the project has not registered any
        project-scoped tools.
    """
    return [name for (pid, name) in _tool_registry_by_project if pid == project_id]


def list_tools() -> list[str]:
    """List all registered tool names.

    Returns:
        List of tool names
    """
    return list(_tool_registry.keys())


__all__ = [
    "register_tool",
    "deregister_tool",
    "discover_all_tools",
    "discover_tools_for_project",
    "get_tool",
    "get_tool_for_project",
    "list_tools",
    "list_project_tools",
    "SecureTool",
    "BaseTool",
]
