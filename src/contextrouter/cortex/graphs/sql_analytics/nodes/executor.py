"""Executor node — runs SQL via registered tool."""

from __future__ import annotations

import json
from decimal import Decimal

from contextcore import get_context_unit_logger
from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    is_debug,
    validate_sql_syntax,
)
from contextrouter.cortex.graphs.sql_analytics.schemas import SqlAnalyticsStateUpdate
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState

logger = get_context_unit_logger(__name__)


def make_execute_node(*, sql_tool: BaseTool | None, tool_names: list[str]):
    """Create the SQL execution node closure.

    Args:
        sql_tool: Resolved SQL tool instance.
        tool_names: Original tool binding names (for error messages).

    Tracing: sql_tool.ainvoke() fires LangChain on_tool_start/end callbacks,
    so BrainAutoTracer captures it automatically — no manual _steps needed.
    """

    async def execute_node(state: SqlAnalyticsState) -> SqlAnalyticsStateUpdate:
        # Lazy resolution: if tool wasn't found at build time, try again now
        resolved_tool = sql_tool
        if not resolved_tool and tool_names:
            from contextrouter.modules.tools import get_tool

            for name in tool_names:
                found = get_tool(name)
                if found:
                    resolved_tool = found
                    logger.info("Lazy-resolved SQL tool: %s", name)
                    break

        if not resolved_tool:
            logger.error("No SQL tool configured! tool_bindings=%s", tool_names)
            return {
                "error": "No SQL tool configured",
                "sql_result": {},
            }

        sql = state.get("sql")
        if not sql:
            logger.warning("execute_node: no SQL in state")
            return {
                "error": "No SQL to execute",
                "sql_result": {},
            }

        # Pre-validate SQL for common syntax errors
        syntax_err = validate_sql_syntax(sql)
        if syntax_err:
            logger.warning("execute_node: pre-validation failed: %s", syntax_err)
            return {
                "error": f"SQL syntax error: {syntax_err}",
            }

        logger.info("execute_node: running SQL tool '%s'", resolved_tool.name)
        if is_debug():
            logger.debug("execute_node: sql=%s", sql[:500])

        try:
            res = await resolved_tool.ainvoke(sql)
            logger.debug("execute_node: tool returned type=%s", type(res).__name__)

            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except Exception:
                    pass

            if not isinstance(res, dict):
                res = {"raw_output": res}

            if res.get("error"):
                logger.warning("execute_node: tool error: %s", res["error"])
                return {"error": res["error"]}

            row_count = len(res.get("rows", []))
            logger.info("execute_node: success, %d rows returned", row_count)
            return {
                "sql_result": _sanitize(res),
                "error": "",
            }

        except Exception as e:
            logger.error("execute_node: exception: %s", e)
            return {"error": str(e)}

    return execute_node


def _sanitize(obj):
    """Recursively convert Decimal → float so protobuf Struct can serialize."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


__all__ = ["make_execute_node"]
