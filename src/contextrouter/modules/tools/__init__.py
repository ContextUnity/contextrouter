"""Tool discovery and registration for dispatcher agent."""

from __future__ import annotations

import asyncio
import inspect
import logging

from langchain_core.tools import BaseTool, tool

from contextrouter.modules.tools.secure import SecureTool

logger = logging.getLogger(__name__)

# Global tool registry — ALL values are SecureTool instances
_tool_registry: dict[str, SecureTool] = {}


def _index_tool_in_brain(tool_instance: BaseTool) -> None:
    """Index tool in Brain Procedural Memory for semantic search.

    This is called automatically when a tool is registered.
    The indexing happens asynchronously and does not block tool registration.

    Args:
        tool_instance: Tool instance to index
    """
    # Brain indexing disabled by default — current implementation triggers
    # full ingestion pipeline (taxonomy, NER, graph) for a placeholder result.
    # Enable via CONTEXT_BRAIN_INDEX_TOOLS=true when PromptGenerator is ready.
    from contextrouter.core import get_core_config

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
    loop.create_task(_async_index_tool(tool_instance))


async def _async_index_tool(tool_instance: BaseTool) -> None:
    """Asynchronously index tool in Brain Procedural Memory.

    Args:
        tool_instance: Tool instance to index
    """
    try:
        import time

        from contextcore import BrainClient

        from contextrouter.core import get_core_config
        from contextrouter.core.brain_token import get_brain_service_token

        brain_endpoint = get_core_config().brain.grpc_endpoint
        brain = BrainClient(host=brain_endpoint, mode="grpc", token=get_brain_service_token())

        # Extract tool information
        tool_name = tool_instance.name
        tool_description = tool_instance.description or ""

        # Get tool schema/parameters if available
        tool_schema = {}
        if hasattr(tool_instance, "args_schema"):
            try:
                schema = tool_instance.args_schema
                if schema:
                    tool_schema = schema.schema() if hasattr(schema, "schema") else {}
            except Exception:
                pass

        # Build content for indexing
        content_parts = [
            f"Tool: {tool_name}",
            f"Description: {tool_description}",
        ]

        if tool_schema:
            properties = tool_schema.get("properties", {})
            if properties:
                content_parts.append("Parameters:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "")
                    content_parts.append(f"  - {param_name} ({param_type}): {param_desc}")

        content = "\n".join(content_parts)

        # Create metadata
        metadata = {
            "tool_name": tool_name,
            "tool_type": "procedural",
            "source_type": "procedural",
            "indexed_at": str(time.time()),
        }

        if tool_schema:
            metadata["tool_schema"] = tool_schema

        # Index in Brain Procedural Memory
        item_id = await brain.upsert(
            tenant_id="default",  # Tools are global, but can be tenant-specific
            content=content,
            source_type="procedural",
            metadata=metadata,
        )
        logger.info("Indexed tool %s in Brain Procedural Memory (id: %s)", tool_name, item_id)

    except Exception as e:
        logger.warning("Failed to index tool %s in Brain: %s", tool_instance.name, e, exc_info=True)


def register_tool(
    tool_instance: BaseTool,
    *,
    permission: str = "",
    scope: str = "read",
    tenant: str = "",
) -> None:
    """Register a tool instance in the global registry.

    If ``tool_instance`` is not already a :class:`SecureTool`, it is
    automatically wrapped with permission enforcement.

    Automatically indexes the tool in Brain Procedural Memory for semantic search.

    Args:
        tool_instance: Tool instance to register.
        permission: Required permission string (default: ``tool:{name}``).
        scope: Required scope — ``read``, ``write``, or ``admin``.
        tenant: Bind tool to a specific tenant/project.  Only tokens
            with this tenant in ``allowed_tenants`` can execute the tool.
    """
    # ── Auto-wrap raw BaseTool → SecureTool ──
    if not isinstance(tool_instance, SecureTool):
        tool_instance = SecureTool.wrap(
            tool_instance,
            permission=permission,
            scope=scope,
            tenant=tenant,
        )
        logger.info(
            "Auto-wrapped tool '%s' → SecureTool (permission=%s, tenant=%s)",
            tool_instance.name,
            tool_instance.required_permission,
            tool_instance.bound_tenant or "(global)",
        )
    else:
        if permission and not tool_instance.required_permission:
            tool_instance.required_permission = permission
        if tenant and not tool_instance.bound_tenant:
            tool_instance.bound_tenant = tenant

    name = tool_instance.name
    if name in _tool_registry:
        logger.info("Tool '%s' already registered, overwriting", name)
    _tool_registry[name] = tool_instance
    logger.debug("Registered tool: %s", name)

    # Index tool in Brain Procedural Memory (async, non-blocking)
    _index_tool_in_brain(tool_instance)


def discover_all_tools() -> list[SecureTool]:
    """Discover all available tools from various sources.

    All returned tools are guaranteed to be :class:`SecureTool` instances
    with mandatory permission enforcement.  Raw ``BaseTool`` instances
    discovered from modules are auto-wrapped.

    Sources:
    1. Registered tools in _tool_registry
    2. Tools from connector modules
    3. Tools from commerce modules
    4. Tools from other registered modules

    Returns:
        List of SecureTool instances
    """
    tools: list[BaseTool] = []

    # NOTE: _tool_registry is populated by module imports below
    # (via register_tool()). We snapshot it at the END of this function.

    # Discover tools from connector modules
    try:
        from contextrouter.modules.connectors import api, file, rss, web

        # Check each connector module for tools
        for module in [api, file, rss, web]:
            for name in dir(module):
                obj = getattr(module, name)
                if inspect.isclass(obj) and issubclass(obj, BaseTool):
                    try:
                        instance = obj()
                        tools.append(instance)
                    except Exception as e:
                        logger.debug("Could not instantiate tool %s: %s", name, e)
    except ImportError as e:
        logger.debug("Could not import connector modules: %s", e)

    # Discover tools from commerce modules
    try:
        from contextrouter.cortex.graphs.commerce import tools as commerce_tools

        # Look for @tool decorated functions
        for name in dir(commerce_tools):
            obj = getattr(commerce_tools, name)
            if inspect.isfunction(obj) and hasattr(obj, "__wrapped__"):
                # Check if it's a tool
                if isinstance(obj, BaseTool):
                    tools.append(obj)
    except ImportError:
        pass

    # Discover tools from news_engine
    try:
        from contextrouter.cortex.graphs.news_engine.agents import language_tool

        if hasattr(language_tool, "apply_language_tool"):
            # Create a tool wrapper for language tool
            @tool
            async def language_correction(text: str, auto_correct: bool = True) -> str:
                """Correct and improve text using language tool.

                Args:
                    text: Text to correct
                    auto_correct: Whether to auto-correct errors

                Returns:
                    Corrected text
                """
                return await language_tool.apply_language_tool(text, auto_correct)

            tools.append(language_correction)
    except ImportError:
        pass

    # Discover sub-agent tools
    try:
        from contextrouter.modules.tools import subagent_tools

        # Tools are auto-registered when module is imported
        # Just ensure the module is loaded
        if hasattr(subagent_tools, "spawn_subagent"):
            logger.debug("Sub-agent tools module loaded")
    except ImportError:
        pass

    # Discover Redis memory tools
    try:
        from contextrouter.modules.tools.redis_memory import (
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
        from contextrouter.modules.tools.brain_memory_tools import (
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
        from contextrouter.modules.tools.sql import SQLToolConfig, create_sql_tools  # noqa: F401

        # SQL tools require project-specific config, so only add if registered
        # Projects register their SQL tools via register_tool()
    except ImportError as e:
        logger.debug("Could not import SQL tools: %s", e)

    # Discover ContextZero privacy tools (optional — requires contextzero package)
    try:
        from contextrouter.modules.tools import privacy_tools  # noqa: F401

        # Tools are auto-registered when module is imported
        logger.debug("ContextZero privacy tools loaded")
    except ImportError:
        logger.debug("ContextZero not installed — privacy tools unavailable")

    # Discover ContextShield security tools (optional — requires contextshield package)
    try:
        from contextrouter.modules.tools import security_tools  # noqa: F401

        # Tools are auto-registered when module is imported
        logger.debug("ContextShield security tools loaded")
    except ImportError:
        logger.debug("ContextShield not installed — security tools unavailable")

    # Discover GCS storage tools (optional — requires google-cloud-storage)
    try:
        from contextrouter.modules.tools import gcs_tools  # noqa: F401

        logger.debug("GCS storage tools loaded")
    except ImportError:
        logger.debug("GCS storage tools unavailable (google-cloud-storage not installed)")

    # Discover Brain trace tools (execution trace logging)
    try:
        from contextrouter.modules.tools import brain_trace_tools  # noqa: F401

        # Tools are auto-registered when module is imported
        logger.debug("Brain trace tools loaded")
    except ImportError as e:
        logger.debug("Could not import Brain trace tools: %s", e)

    # Add all tools that were registered via register_tool()
    # (by module imports above: privacy_tools, security_tools, etc.)
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
            logger.warning(
                "Tool '%s' discovered as raw BaseTool — auto-wrapped to SecureTool",
                t.name,
            )
            secure_tools.append(wrapped)

    logger.info("Discovered %d tools total (all SecureTool)", len(secure_tools))
    return secure_tools


def get_tool(name: str) -> SecureTool | None:
    """Get a tool by name.

    Args:
        name: Tool name

    Returns:
        SecureTool instance or None if not found
    """
    return _tool_registry.get(name)


def deregister_tool(name: str) -> bool:
    """Remove a tool from the global registry by name.

    This is the correct public API for deregistration — callers must
    NOT access ``_tool_registry`` directly.

    Args:
        name: Tool name to deregister.

    Returns:
        True if the tool was found and removed, False if not found.
    """
    if name in _tool_registry:
        del _tool_registry[name]
        logger.info("Deregistered tool: %s", name)
        return True
    logger.debug("Deregister: tool '%s' not found in registry", name)
    return False


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
    "get_tool",
    "list_tools",
    "SecureTool",
    "BaseTool",
    "tool",
]
