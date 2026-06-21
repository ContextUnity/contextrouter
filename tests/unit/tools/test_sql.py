"""Tests for SQL tool validation and execution guards."""

from __future__ import annotations

import time

import pytest

from contextunity.router.modules.tools.sql import SQLToolConfig, execute_sql, validate_sql

pytestmark = pytest.mark.unit


class TestValidateSqlComments:
    def test_mysql_hash_comment_stripped(self) -> None:
        result = validate_sql(
            "SELECT 1 #; DROP TABLE users",
            config=SQLToolConfig(dialect="mysql"),
        )
        assert result.get("valid") is True
        sql = str(result.get("sql", ""))
        assert "DROP" not in sql.upper()

    def test_postgresql_does_not_strip_hash(self) -> None:
        result = validate_sql(
            "SELECT '#not-a-comment' AS x",
            config=SQLToolConfig(dialect="postgresql"),
        )
        assert result.get("valid") is True


class TestExecuteSqlTimeout:
    def test_statement_timeout_ms_enforced(self) -> None:
        def _slow_executor(_sql: str) -> dict[str, object]:
            time.sleep(0.5)
            return {"columns": [], "rows": [], "row_count": 0}

        cfg = SQLToolConfig(
            db_executor=_slow_executor,
            statement_timeout_ms=50,
        )
        start = time.monotonic()
        result = execute_sql("SELECT 1", config=cfg)
        elapsed = time.monotonic() - start

        assert result.get("success") is False
        assert "timed out" in str(result.get("error", "")).lower()
        # The wait must be bounded — returning only after the 0.5s query
        # finished would mean the timeout is ineffective.
        assert elapsed < 0.3
