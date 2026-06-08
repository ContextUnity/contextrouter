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

import asyncio
from typing import TYPE_CHECKING

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk.payload import get_int, get_str, get_str_list
from contextunity.core.types import JsonDict, WireValue, is_json_dict, is_object_dict
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    import asyncio as _asyncio

logger = get_contextunit_logger(__name__)


def create_tool_from_config(
    name: str,
    tool_type: str,
    description: str,
    config: dict[str, WireValue],
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
    if tool_type in ("bidi", "commerce"):
        return _create_bidi_tool(name, description, config)
    from contextunity.core.exceptions import ConfigurationError

    raise ConfigurationError(f"Unsupported tool type: '{tool_type}'.")


def _create_sql_tools(
    name: str,
    description: str,
    config: dict[str, WireValue],
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
    schema_description = get_str(config, "schema_description")
    max_rows = get_int(config, "max_rows", 500)
    statement_timeout_ms = get_int(config, "statement_timeout_ms", 5000)
    forbidden_keywords = get_str_list(config, "forbidden_keywords")
    project_id = get_str(config, "project_id") or get_str(config, "tenant_id")

    # _main_loop is captured lazily on first call (nonlocal in db_executor).
    # db_executor runs in a thread-pool thread (LangChain ainvoke → run_in_executor),
    # so it must schedule async work on the MAIN event loop via run_coroutine_threadsafe.
    _main_loop: _asyncio.AbstractEventLoop | None = None
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    # db_executor — always delegates to BiDi stream
    def db_executor(sql: str) -> JsonDict:
        """Execute SQL via BiDi stream in the project's process.

        DSN never enters the Router process.
        If the stream is not connected, returns an error.
        """
        nonlocal _main_loop
        from contextunity.router.service.stream_executors import get_stream_executor_manager

        manager = get_stream_executor_manager()

        if manager.is_available(project_id, name):
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
                result = future.result(timeout=35)
                if is_json_dict(result):
                    return result
                return {"error": "Stream executor returned non-object SQL payload"}

            result = asyncio.run(manager.execute(project_id, name, {"sql": sql}))
            if is_json_dict(result):
                return result
            return {"error": "Stream executor returned non-object SQL payload"}

        return {
            "error": (
                f"No stream executor available for project '{project_id}'. "
                f"Project must open ToolExecutorStream before executing SQL."
            )
        }

    from contextunity.router.modules.tools.sql import SQLToolConfig, create_sql_tools

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
                object.__setattr__(tool, "description", description)
        elif "validate" in tool.name:
            tool.name = f"{name}_validate"

        tool.tags = (tool.tags or []) + ["federated"]

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


def _create_bidi_tool(
    name: str,
    description: str,
    config: dict[str, WireValue],
) -> list[BaseTool]:
    """Create a BiDi tool that delegates execution via ToolExecutorStream.

    The tool runs entirely in the project's process. Router only forwards
    the call and returns the result.

    Config keys:
        project_id: Project ID for stream routing
    """
    project_id = get_str(config, "project_id") or get_str(config, "tenant_id")

    _main_loop: _asyncio.AbstractEventLoop | None = None
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def bidi_executor(**kwargs: object) -> dict[str, object]:
        """Execute tool via BiDi stream in the project's process."""
        nonlocal _main_loop
        from contextunity.router.service.stream_executors import get_stream_executor_manager

        manager = get_stream_executor_manager()

        if not manager.is_available(project_id, name):
            from langchain_core.tools import ToolException

            raise ToolException(
                (
                    f"No stream executor for project '{project_id}', tool '{name}'. "
                    "Project must open ToolExecutorStream."
                )
            )

        if _main_loop is None:
            try:
                _main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        payload: dict[str, object] = {str(key): value for key, value in kwargs.items()}

        if _main_loop is not None and _main_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                manager.execute(project_id, name, payload),
                _main_loop,
            )
            res = future.result(timeout=120)
        else:
            res = asyncio.run(manager.execute(project_id, name, payload))

        if not is_object_dict(res):
            from langchain_core.tools import ToolException

            raise ToolException(f"Tool '{name}' returned non-object result")

        if "error" in res:
            from langchain_core.tools import ToolException

            raise ToolException(str(res["error"]))
        return {str(key): value for key, value in res.items()}

    from contextunity.router.langchain_boundaries import structured_tool_from_function

    tool = structured_tool_from_function(
        func=bidi_executor,
        name=name,
        description=description or f"BiDi tool: {name}",
        tags=["federated"],
        handle_tool_error=True,
    )

    logger.info(
        "Created BiDi tool '%s' (project=%s)",
        name,
        project_id,
    )
    return [tool]


__all__ = ["create_tool_from_config"]
