"""Unit tests for SQL analytics graph helpers and builder.

Tests extract_json, validate_sql_syntax, acc_tokens, and builder structure
after the monolith→modular refactoring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# SqlAnalyticsState shape
# ---------------------------------------------------------------------------


class TestSqlAnalyticsState:
    """Verify SqlAnalyticsState TypedDict has all required keys."""

    def test_state_has_required_keys(self):
        from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState

        annotations = SqlAnalyticsState.__annotations__
        required = {
            "messages",
            "metadata",
            "sql",
            "purpose",
            "format",
            "sql_result",
            "validation",
            "components",
            "error",
            "retry_count",
            "_start_ts",
            "_token_usage",
            "_steps",
        }
        assert required.issubset(annotations.keys()), (
            f"Missing keys: {required - annotations.keys()}"
        )


# ---------------------------------------------------------------------------
# helpers.extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for robust JSON extraction from LLM text."""

    def test_parses_plain_json(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_in_markdown_block(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        text = '```json\n{"sql": "SELECT 1"}\n```'
        result = extract_json(text)
        assert result == {"sql": "SELECT 1"}

    def test_parses_json_in_plain_code_block(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        text = '```\n{"answer": 42}\n```'
        result = extract_json(text)
        assert result == {"answer": 42}

    def test_extracts_json_with_surrounding_text(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        text = 'Here is the result:\n{"status": "ok"}\nEnd of output.'
        result = extract_json(text)
        assert result == {"status": "ok"}

    def test_returns_empty_dict_for_no_json(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        result = extract_json("No JSON here at all")
        assert result == {}

    def test_returns_empty_dict_for_empty_string(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        result = extract_json("")
        assert result == {}

    def test_handles_nested_json(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import extract_json

        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# helpers.validate_sql_syntax
# ---------------------------------------------------------------------------


class TestValidateSqlSyntax:
    """Tests for SQL syntax pre-validation."""

    def test_valid_sql_returns_none(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import validate_sql_syntax

        assert validate_sql_syntax("SELECT * FROM users WHERE id = 1") is None

    def test_detects_empty_in_clause(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import validate_sql_syntax

        result = validate_sql_syntax("SELECT * FROM users WHERE id IN ()")
        assert result is not None
        assert "empty IN()" in result

    def test_detects_unbalanced_parens(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import validate_sql_syntax

        result = validate_sql_syntax("SELECT * FROM (SELECT id FROM users")
        assert result is not None
        assert "Unbalanced" in result

    def test_balanced_parens_pass(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import validate_sql_syntax

        result = validate_sql_syntax("SELECT * FROM (SELECT id FROM users)")
        assert result is None


# ---------------------------------------------------------------------------
# tools.sql._split_statements (quote-aware splitter)
# ---------------------------------------------------------------------------


class TestSplitStatements:
    """Tests for quote-aware SQL statement splitting."""

    def test_single_statement(self):
        from contextrouter.modules.tools.sql import _split_statements

        result = _split_statements("SELECT 1")
        assert result == ["SELECT 1"]

    def test_multiple_statements(self):
        from contextrouter.modules.tools.sql import _split_statements

        result = _split_statements("SELECT 1; SELECT 2")
        assert result == ["SELECT 1", "SELECT 2"]

    def test_semicolon_inside_single_quotes(self):
        from contextrouter.modules.tools.sql import _split_statements

        sql = "SELECT STRING_AGG(x, '; ') FROM t"
        result = _split_statements(sql)
        assert len(result) == 1
        assert result[0] == sql

    def test_semicolon_inside_double_quotes(self):
        from contextrouter.modules.tools.sql import _split_statements

        sql = 'SELECT "col;name" FROM t'
        result = _split_statements(sql)
        assert len(result) == 1

    def test_doubled_quote_escape(self):
        from contextrouter.modules.tools.sql import _split_statements

        sql = "SELECT 'it''s; here' FROM t"
        result = _split_statements(sql)
        assert len(result) == 1

    def test_mixed_quotes_and_real_separator(self):
        from contextrouter.modules.tools.sql import _split_statements

        sql = "SELECT STRING_AGG(x, '; ') FROM t; SELECT 2"
        result = _split_statements(sql)
        assert len(result) == 2
        assert "STRING_AGG" in result[0]
        assert result[1] == "SELECT 2"


# ---------------------------------------------------------------------------
# tools.sql.validate_sql (semicolons in string literals)
# ---------------------------------------------------------------------------


class TestValidateSql:
    """Tests for the SQL validation function."""

    def test_valid_select(self):
        from contextrouter.modules.tools.sql import validate_sql

        result = validate_sql("SELECT * FROM users LIMIT 10")
        assert result["valid"] is True

    def test_valid_with_cte(self):
        from contextrouter.modules.tools.sql import validate_sql

        result = validate_sql("WITH cte AS (SELECT 1) SELECT * FROM cte LIMIT 10")
        assert result["valid"] is True

    def test_rejects_insert(self):
        from contextrouter.modules.tools.sql import validate_sql

        result = validate_sql("INSERT INTO users VALUES (1)")
        assert result["valid"] is False

    def test_semicolon_in_string_literal_does_not_break_validation(self):
        """Regression: STRING_AGG(x, '; ') was incorrectly splitting on ';'."""
        from contextrouter.modules.tools.sql import validate_sql

        sql = (
            "SELECT dp.name, STRING_AGG(DISTINCT mr.error_comment, '; ' "
            "ORDER BY mr.error_comment) FILTER (WHERE mr.error_comment IS NOT NULL) "
            "FROM medical_records mr JOIN departments dp ON mr.department_id = dp.id "
            "GROUP BY dp.name LIMIT 100"
        )
        result = validate_sql(sql)
        assert result["valid"] is True, f"Expected valid but got: {result.get('error')}"


# ---------------------------------------------------------------------------
# helpers.acc_tokens
# ---------------------------------------------------------------------------


class TestAccTokens:
    """Tests for token usage accumulation."""

    def test_accumulates_from_empty(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import acc_tokens

        state: dict = {}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_cost": 0.01}
        result = acc_tokens(state, usage)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_cost"] == pytest.approx(0.01)

    def test_accumulates_with_previous(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import acc_tokens

        state = {"_token_usage": {"input_tokens": 50, "output_tokens": 25, "total_cost": 0.005}}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_cost": 0.01}
        result = acc_tokens(state, usage)
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 75
        assert result["total_cost"] == pytest.approx(0.015)

    def test_handles_empty_usage(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import acc_tokens

        state = {"_token_usage": {"input_tokens": 50, "output_tokens": 25, "total_cost": 0.005}}
        result = acc_tokens(state, {})
        assert result["input_tokens"] == 50


# ---------------------------------------------------------------------------
# Builder structure
# ---------------------------------------------------------------------------


class TestSqlAnalyticsBuilder:
    """Test sql_analytics builder with minimal config."""

    @patch("contextrouter.cortex.graphs.sql_analytics.builder.discover_all_tools")
    def test_build_returns_compiled_graph(self, mock_tools):
        from contextrouter.cortex.graphs.sql_analytics.builder import build_sql_analytics_graph

        mock_tools.return_value = []
        config = {"tool_bindings": [], "model_key": None}
        compiled = build_sql_analytics_graph(config)
        # A compiled graph should have invoke or ainvoke
        assert hasattr(compiled, "ainvoke") or hasattr(compiled, "invoke")

    @patch("contextrouter.cortex.graphs.sql_analytics.builder.discover_all_tools")
    def test_build_with_sql_tool(self, mock_tools):
        from contextrouter.cortex.graphs.sql_analytics.builder import build_sql_analytics_graph

        tool = MagicMock()
        tool.name = "nszu_sql"
        mock_tools.return_value = [tool]

        config = {"tool_bindings": ["nszu_sql"], "model_key": None}
        compiled = build_sql_analytics_graph(config)
        assert compiled is not None


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestSqlAnalyticsExports:
    """Ensure all exports are properly defined."""

    def test_helpers_all_exports(self):
        from contextrouter.cortex.graphs.sql_analytics.helpers import __all__

        expected = {
            "StepTimer",
            "acc_tokens",
            "extract_json",
            "invoke_model",
            "is_debug",
            "step",
            "validate_sql_syntax",
        }
        assert set(__all__) == expected

    def test_state_exports(self):
        from contextrouter.cortex.graphs.sql_analytics.state import __all__

        assert "SqlAnalyticsState" in __all__

    def test_builder_exports(self):
        from contextrouter.cortex.graphs.sql_analytics.builder import __all__

        assert "build_sql_analytics_graph" in __all__
