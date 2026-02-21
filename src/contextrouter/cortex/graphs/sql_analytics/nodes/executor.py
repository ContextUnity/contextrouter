"""Executor node — runs SQL via registered tool."""

from __future__ import annotations

import json
import logging
from decimal import Decimal

from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    StepTimer,
    is_debug,
    step,
    validate_sql_syntax,
)
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState

logger = logging.getLogger(__name__)


def make_execute_node(*, sql_tool: BaseTool | None, tool_names: list[str]):
    """Create the SQL execution node closure.

    Args:
        sql_tool: Resolved SQL tool instance.
        tool_names: Original tool binding names (for error messages).
    """

    async def execute_node(state: SqlAnalyticsState):
        if not sql_tool:
            logger.error("No SQL tool configured! tool_bindings=%s", tool_names)
            return {
                "error": "No SQL tool configured",
                "sql_result": {},
                "_steps": [step("execute_sql", status="error", reason="no_tool")],
            }

        sql = state.get("sql")
        if not sql:
            logger.warning("execute_node: no SQL in state")
            return {
                "error": "No SQL to execute",
                "sql_result": {},
                "_steps": [step("execute_sql", status="error", reason="no_sql")],
            }

        # Pre-validate SQL for common syntax errors
        syntax_err = validate_sql_syntax(sql)
        if syntax_err:
            logger.warning("execute_node: pre-validation failed: %s", syntax_err)
            return {
                "error": f"SQL syntax error: {syntax_err}",
                "_steps": [
                    step(
                        "execute_sql",
                        status="error",
                        reason="syntax",
                        request={"sql": sql[:500]},
                    )
                ],
            }

        logger.info("execute_node: running SQL tool '%s'", sql_tool.name)
        if is_debug():
            logger.debug("execute_node: sql=%s", sql[:500])

        timer = StepTimer()
        try:
            with timer:
                res = await sql_tool.ainvoke(sql)
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
                return {
                    "error": res["error"],
                    "_steps": [
                        step(
                            "execute_sql",
                            status="error",
                            timer=timer,
                            request={"sql": sql[:500]},
                            result={"error": str(res["error"])[:500]},
                        )
                    ],
                }

            row_count = len(res.get("rows", []))
            logger.info("execute_node: success, %d rows returned", row_count)
            return {
                "sql_result": _sanitize(res),
                "error": "",
                "_steps": [
                    step(
                        "execute_sql",
                        timer=timer,
                        request={"sql": sql[:500]},
                        result={
                            "row_count": row_count,
                            "columns": res.get("columns", []),
                        },
                        row_count=row_count,
                    )
                ],
            }

        except Exception as e:
            logger.error("execute_node: exception: %s", e)
            return {
                "error": str(e),
                "_steps": [
                    step(
                        "execute_sql",
                        status="error",
                        timer=timer,
                        request={"sql": sql[:500]},
                    )
                ],
            }

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
