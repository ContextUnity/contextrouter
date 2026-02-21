"""Agnostic SQL tools for LangGraph-based agents.

These tools are project-agnostic — they do NOT import Django, psycopg2,
or any project-specific code. Instead, they accept callbacks for:
- DB execution (project provides its own connection)
- Schema info (project provides table descriptions)

Usage in a graph:
    from contextrouter.modules.tools.sql import (
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

import logging
import re
import time
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


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
    db_executor: Callable[..., dict[str, Any]] | None = None
    dialect: str = "postgresql"
    max_rows: int = 500
    statement_timeout_ms: int = 5000
    forbidden_keywords: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── Validation (deterministic, no LLM) ──────────────────────────────

_BUILTIN_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXECUTE|COPY|INTO)\b",
    re.IGNORECASE,
)

_FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file"
    r"|lo_import|lo_export|lo_get|lo_put"
    r"|dblink|dblink_exec|dblink_connect"
    r"|pg_sleep|set_config"
    r"|current_setting\s*\(\s*['\"]superuser)",
    re.IGNORECASE,
)


def _split_statements(sql: str) -> list[str]:
    """Split SQL into statements on `;`, respecting quoted strings.

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


def validate_sql(sql: str, *, config: SQLToolConfig | None = None) -> dict[str, Any]:
    """Validate and sanitize SQL.  Return ``{"valid": True, "sql": cleaned}``
    or ``{"valid": False, "error": reason}``.

    Deterministic — no LLM calls.
    """
    cfg = config or SQLToolConfig()
    raw = (sql or "").strip()
    if not raw:
        return {"valid": False, "error": "Empty SQL"}

    # Strip comments
    clean = re.sub(r"/\*.*?\*/", " ", raw, flags=re.DOTALL)
    clean = re.sub(r"--[^\n]*", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(";").strip()

    if not clean:
        return {"valid": False, "error": "Empty SQL after stripping comments"}

    # Multiple statements → take last (quote-aware: don't split on ';' inside strings)
    statements = _split_statements(clean)
    if len(statements) > 1:
        clean = statements[-1]

    # Must start with SELECT / WITH
    first_word = clean.split()[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return {
            "valid": False,
            "error": f"Query must start with SELECT or WITH, got: {first_word}",
        }

    # Forbidden keywords
    match = _BUILTIN_FORBIDDEN.search(clean)
    if match:
        return {"valid": False, "error": f"Forbidden keyword: {match.group()}"}

    # Forbidden functions
    func_match = _FORBIDDEN_FUNCTIONS.search(clean)
    if func_match:
        return {"valid": False, "error": f"Forbidden function: {func_match.group()}"}

    for kw in cfg.forbidden_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", clean, re.IGNORECASE):
            return {"valid": False, "error": f"Forbidden keyword: {kw}"}

    # Fix LIMIT 0
    clean = re.sub(r"\bLIMIT\s+0\b", f"LIMIT {cfg.max_rows}", clean, flags=re.IGNORECASE)

    # Add LIMIT if missing
    if "LIMIT" not in clean.upper():
        clean = f"{clean} LIMIT {cfg.max_rows}"

    return {"valid": True, "sql": clean}


def execute_sql(sql: str, *, config: SQLToolConfig) -> dict[str, Any]:
    """Execute validated SQL via the project-provided ``db_executor``.

    Returns dict with ``columns``, ``rows``, ``row_count``, ``elapsed_ms``
    or ``{"error": "..."}`` on failure.
    """
    if config.db_executor is None:
        return {"error": "No db_executor configured"}

    # Validate first
    result = validate_sql(sql, config=config)
    if not result.get("valid"):
        return {"error": result.get("error", "Validation failed")}

    clean_sql = result["sql"]

    start = time.monotonic()
    try:
        data = config.db_executor(clean_sql)
    except Exception as e:
        logger.warning("SQL execution failed: %s | SQL: %.100s…", e, clean_sql)
        return {"error": f"SQL execution error: {e}"}

    elapsed = (time.monotonic() - start) * 1000

    return {
        "columns": data.get("columns", []),
        "rows": data.get("rows", []),
        "row_count": data.get("row_count", len(data.get("rows", []))),
        "elapsed_ms": round(elapsed, 1),
        "sql_executed": clean_sql,
    }


# ── LangChain Tool wrappers ─────────────────────────────────────────


def create_sql_tools(config: SQLToolConfig) -> list:
    """Create LangChain-compatible SQL tools bound to a specific config.

    Returns list of tool instances ready for ``llm.bind_tools(tools)``.
    """
    from langchain_core.tools import tool

    @tool
    def validate_sql_tool(sql: str) -> dict:
        """Validate a SQL query for safety. Returns {valid, sql} or {valid, error}."""
        return validate_sql(sql, config=config)

    @tool
    def execute_sql_tool(sql: str) -> dict:
        """Execute a validated read-only SQL query against the database.
        Returns {columns, rows, row_count, elapsed_ms} or {error}.
        """
        return execute_sql(sql, config=config)

    return [validate_sql_tool, execute_sql_tool]


# ── Column label translation helper ─────────────────────────────────

DEFAULT_COLUMN_LABELS: dict[str, str] = {
    "doctor_name": "Лікар",
    "department_name": "Відділення",
    "rejections": "Кількість відмов",
    "rejection_count": "Кількість відмов",
    "package_number": "Пакет",
    "package_name": "Назва пакету",
    "tariff": "Тариф (грн)",
    "total_tariff": "Сума тарифу (грн)",
    "total_loss": "Втрати (грн)",
    "inclusion_status": "Статус",
    "diagnosis_code": "Код діагнозу",
    "diagnosis_name": "Діагноз",
    "patient_code": "Код пацієнта",
    "doctor_id": "ID лікаря",
    "department_id": "ID відділення",
    "cnt": "Кількість",
    "count": "Кількість",
    "doctor_count": "Кількість лікарів",
    "patient_records_count": "Кількість записів",
    "error_comment": "Причина відхилення",
}

HIDDEN_COLUMNS = {"rn", "row_number", "rank", "sub", "row_num"}


def humanize_columns(
    query_result: dict[str, Any],
    *,
    labels: dict[str, str] | None = None,
    hidden: set[str] | None = None,
) -> dict[str, Any]:
    """Translate column keys to human labels and hide service columns.

    Args:
        query_result: Dict with 'columns' and 'rows' keys.
        labels: Column key → label mapping. Defaults to DEFAULT_COLUMN_LABELS.
        hidden: Column keys to hide. Defaults to HIDDEN_COLUMNS.

    Returns:
        Dict with translated 'columns' list and filtered 'rows'.
    """
    _labels = labels or DEFAULT_COLUMN_LABELS
    _hidden = hidden or HIDDEN_COLUMNS

    columns = [
        {"key": c, "label": _labels.get(c, c)}
        for c in query_result.get("columns", [])
        if c not in _hidden
    ]
    visible_keys = {col["key"] for col in columns}
    rows = [
        {k: v for k, v in row.items() if k in visible_keys} for row in query_result.get("rows", [])
    ]

    return {
        **query_result,
        "columns": columns,
        "rows": rows,
    }


__all__ = [
    "SQLToolConfig",
    "validate_sql",
    "execute_sql",
    "create_sql_tools",
    "humanize_columns",
    "DEFAULT_COLUMN_LABELS",
    "HIDDEN_COLUMNS",
]
