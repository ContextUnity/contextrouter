"""Factory for creating BaseTool instances from registration config.

Converts project-provided tool config into real LangChain BaseTool instances
that execute inside Router's process.

Supported tool types:
- ``sql``: creates SQL executor that delegates via BiDi stream to the project.
  The project process executes SQL locally and returns results.
  If no stream is connected, returns an error — there is no fallback.

Security model:
  DSN never leaves the project process. Router only forwards SQL strings
  and receives result rows via the ToolExecutorStream.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def create_tool_from_config(
    name: str,
    tool_type: str,
    description: str,
    config: dict[str, Any],
) -> list[BaseTool]:
    """Create BaseTool instance(s) from registration config.

    Args:
        name: Tool name
        tool_type: Tool type (sql, search, custom)
        description: Human-readable description
        config: Type-specific config dict

    Returns:
        List of BaseTool instances

    Raises:
        ValueError: Unknown tool type or missing config
    """
    if tool_type == "sql":
        return _create_sql_tools(name, description, config)
    else:
        raise ValueError(f"Unknown tool type: {tool_type}. Supported: sql")


def _create_sql_tools(
    name: str,
    description: str,
    config: dict[str, Any],
) -> list[BaseTool]:
    """Create SQL tools that delegate execution via BiDi stream.

    SQL is always executed in the project's process via ToolExecutorStream.
    If the stream is not connected, the tool returns an error.
    DSN never enters the Router process.

    Config keys:
        tenant_id: Tenant ID for stream routing
        project_id: Project ID for stream routing (falls back to tenant_id)
        schema_description: DB schema for LLM prompt
        read_only: bool (default True) — SELECT only enforcement is in project
        max_rows: int (default 500)
        statement_timeout_ms: int (default 5000)
        forbidden_keywords: list[str]
    """
    schema_description = config.get("schema_description", "")
    max_rows = config.get("max_rows", 500)
    statement_timeout_ms = config.get("statement_timeout_ms", 5000)
    forbidden_keywords = config.get("forbidden_keywords", [])
    project_id = config.get("project_id", config.get("tenant_id", ""))

    # _main_loop is captured lazily on first call (nonlocal in db_executor).
    # db_executor runs in a thread-pool thread (LangChain ainvoke → run_in_executor),
    # so it must schedule async work on the MAIN event loop via run_coroutine_threadsafe.
    _main_loop: "_asyncio.AbstractEventLoop | None" = None
    try:
        import asyncio as _asyncio

        _main_loop = _asyncio.get_running_loop()
    except RuntimeError:
        pass

    # db_executor — always delegates to BiDi stream
    def db_executor(sql: str) -> dict[str, Any]:
        """Execute SQL via BiDi stream in the project's process.

        DSN never enters the Router process.
        If the stream is not connected, returns an error.
        """
        nonlocal _main_loop
        from contextrouter.service.stream_executors import get_stream_executor_manager

        manager = get_stream_executor_manager()

        if manager.is_available(project_id, name):
            import asyncio

            if _main_loop is None:
                try:
                    _main_loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

            if _main_loop is not None and _main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    manager.execute(project_id, name, {"sql": sql}),
                    _main_loop,
                )
                return future.result(timeout=35)
            else:
                return asyncio.run(manager.execute(project_id, name, {"sql": sql}))

        return {
            "error": (
                f"No stream executor available for project '{project_id}'. "
                f"Project must open ToolExecutorStream before executing SQL."
            )
        }

    from contextrouter.modules.tools.sql import SQLToolConfig, create_sql_tools

    sql_config = SQLToolConfig(
        schema_description=schema_description,
        db_executor=db_executor,
        dialect="postgresql",
        max_rows=max_rows,
        statement_timeout_ms=statement_timeout_ms,
        forbidden_keywords=forbidden_keywords,
    )

    tools = create_sql_tools(sql_config)

    # Override tool names/descriptions if provided
    for tool in tools:
        if "execute" in tool.name:
            tool.name = name
            if description:
                tool.description = description
        elif "validate" in tool.name:
            tool.name = f"{name}_validate"

    mode = "stream-only"
    logger.info(
        "Created SQL tools for '%s' (mode=%s, project=%s, max_rows=%d, timeout=%dms)",
        name,
        mode,
        project_id,
        max_rows,
        statement_timeout_ms,
    )
    return tools


__all__ = ["create_tool_from_config"]
