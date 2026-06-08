"""Tests for LangChain vendor boundary helpers."""

from __future__ import annotations

from contextunity.router.langchain_boundaries import structured_tool_from_function


def test_structured_tool_from_function_infers_args_schema() -> None:
    def run_query(sql: str) -> str:
        """Execute a SQL query."""
        return sql

    tool = structured_tool_from_function(
        func=run_query,
        name="medical_sql",
        description="Run SQL against the medical database",
        tags=["federated"],
        handle_tool_error=True,
    )

    assert tool.name == "medical_sql"
    assert tool.args_schema is not None
