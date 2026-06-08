"""Executor node — runs SQL via registered tool."""

from __future__ import annotations

from decimal import Decimal
from typing import ClassVar, Protocol, TypeGuard

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.types import is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.types import GraphState, StateUpdate

from ...state_routing import read_state_str
from .sql import SqlResultDict, is_debug, validate_sql_syntax


class SqlExecutorConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(default=120, ge=1, le=600)
    max_rows: int = Field(default=1000, ge=1, le=50000)


logger = get_contextunit_logger(__name__)


class _SqlTool(Protocol):
    """Runtime surface used by the SQL executor."""

    name: str

    async def ainvoke(self, input: object) -> object: ...


def _is_sql_tool(value: object) -> TypeGuard[_SqlTool]:
    """Runtime guard for SQL tool instances."""
    tool_name = getattr(value, "name", None)
    return isinstance(tool_name, str) and callable(getattr(value, "ainvoke", None))


def make_execute_node(*, sql_tool: _SqlTool | None, tool_names: list[str]):
    """Create the SQL execution node closure.

    Args:
        sql_tool: Resolved SQL tool instance.
        tool_names: Original tool binding names (for error messages).

    Tracing: sql_tool.ainvoke() fires LangChain on_tool_start/end callbacks,
    so BrainAutoTracer captures it automatically — no manual _steps needed.
    """

    async def execute_node(state: GraphState) -> StateUpdate:
        """Execute SQL via the resolved tool with pre-validation.

        Reads ``sql`` from state, validates syntax, invokes the tool,
        and normalizes the result into a ``SqlResultDict``.
        Performs lazy tool resolution if the tool was unavailable at
        graph build time.

        Args:
            state: Graph state containing ``sql`` field.

        Returns:
            State update with ``sql_result`` and ``error`` in ``dynamic``.
        """
        # Lazy resolution: if tool wasn't found at build time, try again now
        resolved_tool = sql_tool
        if not resolved_tool and tool_names:
            from contextunity.router.cortex.config_resolution import metadata_project_id
            from contextunity.router.modules.tools import get_tool_for_project

            for name in tool_names:
                found = get_tool_for_project(metadata_project_id(state), name)
                if _is_sql_tool(found):
                    resolved_tool = found
                    logger.info("Lazy-resolved SQL tool: %s", name)
                    break

        if not resolved_tool:
            logger.error("No SQL tool configured! tool_bindings=%s", tool_names)
            return {
                "dynamic": {"error": "No SQL tool configured", "sql_result": {}},
            }

        sql = read_state_str(state, "sql")
        if not sql:
            logger.warning("execute_node: no SQL in state")
            return {
                "dynamic": {"error": "No SQL to execute", "sql_result": {}},
            }

        # Pre-validate SQL for common syntax errors
        syntax_err = validate_sql_syntax(sql)
        if syntax_err:
            logger.warning("execute_node: pre-validation failed: %s", syntax_err)
            return {
                "dynamic": {"error": f"SQL syntax error: {syntax_err}"},
            }

        logger.info("execute_node: running SQL tool '%s'", resolved_tool.name)
        if is_debug():
            logger.debug("execute_node: sql=%s", sql[:500])

        try:
            res = await resolved_tool.ainvoke(sql)
            logger.debug("execute_node: tool returned type=%s", type(res).__name__)

            if isinstance(res, str):
                try:
                    res = json_loads(res)
                except Exception:  # graceful-degrade: SQL error returns empty result
                    pass

            if not is_object_dict(res):
                res = {"raw_output": res}
            else:
                res = {str(key): value for key, value in res.items()}

            if res.get("error"):
                logger.warning("execute_node: tool error: %s", res["error"])
                return {"dynamic": {"error": res["error"]}}

            rows = res.get("rows", [])
            row_count = len(rows) if is_object_list(rows) else 0
            logger.info("execute_node: success, %d rows returned", row_count)
            sanitized = _sanitize(res)
            sql_result: SqlResultDict
            if is_object_dict(sanitized):
                sane: dict[str, object] = {str(key): value for key, value in sanitized.items()}
                raw_rows = sane.get("rows", [])
                raw_cols = sane.get("columns", [])
                typed_rows: list[dict[str, object]] = (
                    [
                        {str(key): value for key, value in row.items()}
                        for row in raw_rows
                        if is_object_dict(row)
                    ]
                    if is_object_list(raw_rows)
                    else []
                )
                typed_cols = (
                    [str(column) for column in raw_cols] if is_object_list(raw_cols) else []
                )
                sql_result = {
                    "rows": typed_rows,
                    "columns": typed_cols,
                    "row_count": row_count,
                }
                dur = sane.get("duration_ms")
                if isinstance(dur, (int, float)):
                    sql_result["duration_ms"] = float(dur)
            else:
                sql_result = {"rows": [], "columns": [], "row_count": 0}
            return {
                "dynamic": {"sql_result": sql_result, "error": ""},
            }

        except Exception as e:  # graceful-degrade: SQL error returns empty result
            logger.error("execute_node: exception: %s", e)
            return {"dynamic": {"error": str(e)}}

    return execute_node


def _sanitize(obj: object) -> object:
    """Recursively convert ``Decimal`` values to ``float`` for protobuf Struct.

    Args:
        obj: Arbitrary nested structure (dict, list, or scalar).

    Returns:
        Same structure with ``Decimal`` instances replaced by ``float``.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    if is_object_dict(obj):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if is_object_list(obj):
        return [_sanitize(value) for value in obj]
    return obj


__all__ = ["make_execute_node"]
