"""Agnostic SQL tools for LangGraph-based agents.

These tools are project-agnostic — they do NOT import Django, psycopg2,
or any project-specific code. Instead, they accept callbacks for:
- DB execution (project provides its own connection)
- Schema info (project provides table descriptions)

Defense-in-depth: the regex blocklist here is **not** sufficient on its own.
Hosts should also use a read-only database role and a server-side
``statement_timeout`` (PostgreSQL) or equivalent engine limits.

Usage in a graph:
    from contextunity.router.modules.tools.sql import (
        create_sql_tools,
        SQLToolConfig,
    )

    cfg = SQLToolConfig(
        schema_description="... tables ...",
        db_executor=my_execute_fn,
        dialect="postgresql",
    )
    tools = create_sql_tools(cfg)
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from typing import TYPE_CHECKING, Callable, ClassVar

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonDict, JsonValue, is_json_dict, is_json_value
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.langchain_boundaries import tool
from contextunity.router.modules.tools.schemas import SQLResult

logger = get_contextunit_logger(__name__)


# ── Configuration ────────────────────────────────────────────────────


class SQLToolConfig(BaseModel):
    """Configuration for SQL tools.

    Attributes:
        schema_description: Human-readable DB schema for LLM prompt.
        db_executor: Callback ``(sql: str) -> dict`` with keys
            ``columns``, ``rows``, ``row_count``, ``elapsed_ms``.
        dialect: SQL dialect hint for LLM (postgresql, mysql, sqlite).
        max_rows: Maximum rows to return.
        statement_timeout_ms: Max query time.
        forbidden_keywords: Additional SQL keywords to block.
    """

    schema_description: str = ""
    db_executor: Callable[[str], JsonDict] | None = None
    dialect: str = "postgresql"
    max_rows: int = 500
    statement_timeout_ms: int = 5000
    forbidden_keywords: list[str] = Field(default_factory=list)

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)


# ── Validation (deterministic, no LLM) ──────────────────────────────

_BUILTIN_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXECUTE|COPY|INTO)\b",
    re.IGNORECASE,
)

_FORBIDDEN_FUNCTIONS = re.compile(
    (
        r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file"
        r"|lo_import|lo_export|lo_get|lo_put"
        r"|dblink|dblink_exec|dblink_connect"
        r"|pg_sleep|set_config"
        r"|current_setting\s*\(\s*['\"]superuser)"
    ),
    re.IGNORECASE,
)


def _split_statements(sql: str) -> list[str]:
    """Split SQL into statements on ``;``, respecting quoted strings.

    Handles single-quotes, double-quotes, and doubled-quote escapes
    (e.g. ``'it''s'``).  Does NOT split on ``;`` inside string literals.
    """
    statements: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_quote:
            current.append(ch)
            if ch == in_quote:
                # Check for doubled-quote escape (e.g. '')
                if i + 1 < len(sql) and sql[i + 1] == in_quote:
                    current.append(sql[i + 1])
                    i += 2
                    continue
                in_quote = None
        elif ch in ("'", '"'):
            in_quote = ch
            current.append(ch)
        elif ch == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1
    stmt = "".join(current).strip()
    if stmt:
        statements.append(stmt)
    return statements


def _strip_sql_comments(sql: str, *, dialect: str) -> str:
    """Strip block/line comments; MySQL ``#`` lines only when *dialect* is mysql."""
    clean = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    clean = re.sub(r"--[^\n]*", " ", clean)
    if dialect.strip().lower() == "mysql":
        clean = re.sub(r"#[^\n]*", " ", clean)
    return re.sub(r"\s+", " ", clean).strip().rstrip(";").strip()


def _call_db_executor(
    executor: Callable[[str], JsonDict],
    sql: str,
    *,
    timeout_ms: int,
) -> JsonDict:
    """Run *executor* with a client-side timeout guard."""
    if timeout_ms <= 0:
        return executor(sql)

    # NOTE: a context-managed pool calls shutdown(wait=True) on exit and would
    # block until the runaway query finishes — defeating the timeout. Manage the
    # pool manually and shut down with wait=False so the caller is freed
    # immediately. The orphaned worker keeps its DB connection until the query
    # ends, so a server-side statement_timeout remains the real protection.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(executor, sql)
    try:
        result = future.result(timeout=timeout_ms / 1000.0)
    except concurrent.futures.TimeoutError as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"SQL execution timed out after {timeout_ms}ms") from exc
    pool.shutdown(wait=False)

    if not is_json_dict(result):
        raise TypeError("db_executor returned non-object payload")
    return result


def validate_sql(sql: str, *, config: SQLToolConfig | None = None) -> SQLResult:
    """Validate and sanitize SQL.  Return ``{"valid": True, "sql": cleaned}``
    or ``{"valid": False, "error": reason}``.

    Deterministic — no LLM calls.
    """
    cfg = config or SQLToolConfig()
    raw = (sql or "").strip()
    if not raw:
        return SQLResult(success=False, valid=False, error="Empty SQL")

    # Strip comments (dialect-aware for MySQL ``#``)
    clean = _strip_sql_comments(raw, dialect=cfg.dialect)

    if not clean:
        return SQLResult(success=False, valid=False, error="Empty SQL after stripping comments")

    # Multiple statements → take last (quote-aware: don't split on ';' inside strings)
    statements = _split_statements(clean)
    if len(statements) > 1:
        clean = statements[-1]

    # Must start with SELECT / WITH
    first_word = clean.split()[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return SQLResult(
            success=False,
            valid=False,
            error=f"Query must start with SELECT or WITH, got: {first_word}",
        )

    # Forbidden keywords
    match = _BUILTIN_FORBIDDEN.search(clean)
    if match:
        return SQLResult(success=False, valid=False, error=f"Forbidden keyword: {match.group()}")

    # Forbidden functions
    func_match = _FORBIDDEN_FUNCTIONS.search(clean)
    if func_match:
        return SQLResult(
            success=False, valid=False, error=f"Forbidden function: {func_match.group()}"
        )

    for kw in cfg.forbidden_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", clean, re.IGNORECASE):
            return SQLResult(success=False, valid=False, error=f"Forbidden keyword: {kw}")

    # Fix LIMIT 0
    clean = re.sub(r"\bLIMIT\s+0\b", f"LIMIT {cfg.max_rows}", clean, flags=re.IGNORECASE)

    # Add LIMIT if missing
    if "LIMIT" not in clean.upper():
        clean = f"{clean} LIMIT {cfg.max_rows}"

    return SQLResult(success=True, valid=True, sql=clean)


def execute_sql(sql: str, *, config: SQLToolConfig) -> SQLResult:
    """Execute validated SQL via the project-provided ``db_executor``.

    Returns dict with ``columns``, ``rows``, ``row_count``, ``elapsed_ms``
    or ``{"error": "..."}`` on failure.
    """
    if config.db_executor is None:
        return SQLResult(success=False, error="No db_executor configured")

    # Validate first
    result = validate_sql(sql, config=config)
    if not result.get("valid"):
        return SQLResult(success=False, error=result.get("error", "Validation failed"))

    clean_sql = result.get("sql", "")

    start = time.monotonic()
    try:
        data = _call_db_executor(
            config.db_executor,
            clean_sql,
            timeout_ms=config.statement_timeout_ms,
        )
    except TimeoutError as e:
        logger.warning("SQL execution timed out: %s | SQL: %.100s…", e, clean_sql)
        return SQLResult(success=False, error=str(e))
    except Exception as e:
        logger.warning("SQL execution failed: %s | SQL: %.100s…", e, clean_sql)
        return SQLResult(success=False, error=f"SQL execution error: {e}")

    if not is_json_dict(data):
        return SQLResult(success=False, error="db_executor returned non-object payload")

    error_raw = data.get("error")
    if error_raw is not None:
        logger.warning("SQL execution error: %s | SQL: %.100s…", error_raw, clean_sql)
        return SQLResult(success=False, error=str(error_raw))

    elapsed = (time.monotonic() - start) * 1000

    columns_raw = data.get("columns", [])
    columns = [str(col) for col in columns_raw] if isinstance(columns_raw, list) else []

    rows_raw = data.get("rows", [])
    rows: list[list[JsonValue]] = []
    if isinstance(rows_raw, list):
        for row in rows_raw:
            if not isinstance(row, list):
                continue
            row_out: list[JsonValue] = []
            for cell in row:
                if is_json_value(cell):
                    row_out.append(cell)
            rows.append(row_out)

    row_count_raw = data.get("row_count")
    row_count = int(row_count_raw) if isinstance(row_count_raw, (int, float)) else len(rows)

    return SQLResult(
        success=True,
        columns=columns,
        rows=rows,
        row_count=row_count,
        elapsed_ms=round(elapsed, 1),
        sql_executed=clean_sql,
    )


# ── LangChain Tool wrappers ─────────────────────────────────────────


def create_sql_tools(config: SQLToolConfig) -> list[BaseTool]:
    """Create LangChain-compatible SQL tools bound to a specific config.

    Returns list of tool instances ready for ``llm.bind_tools(tools)``.
    """
    from langchain_core.tools import ToolException

    @tool
    def validate_sql_tool(sql: str) -> SQLResult:
        """Validate a SQL query for safety. Returns {valid, sql} or {valid, error}."""
        res = validate_sql(sql, config=config)
        if not res.get("valid"):
            raise ToolException(res.get("error", "Validation failed"))
        return res

    @tool
    def execute_sql_tool(sql: str) -> SQLResult:
        """Execute a validated read-only SQL query against the database.
        Returns {columns, rows, row_count, elapsed_ms} or {error}.
        """
        res = execute_sql(sql, config=config)
        if "error" in res:
            raise ToolException(res["error"])
        return res

    # @tool decorator doesn't accept handle_tool_error kwarg —
    # set it on the instances so ToolException is returned as text
    # to the LLM agent instead of crashing the graph.
    validate_sql_tool.handle_tool_error = True
    execute_sql_tool.handle_tool_error = True

    return [validate_sql_tool, execute_sql_tool]


__all__ = [
    "SQLToolConfig",
    "validate_sql",
    "execute_sql",
    "create_sql_tools",
]
