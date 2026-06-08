"""Adaptive output formatter — shapes final response based on data types.

Analyzes intermediate_results to determine optimal output format:
- Tabular data → structured table/chart representation
- Text content → markdown with citations
- Mixed → combined format with clear sections

This is the final output shaping step before reflect/end.
"""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonDict, is_json_dict, is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.types import GraphState, StateUpdate

logger = get_contextunit_logger(__name__)


class FormatterConfig(BaseModel, frozen=True):
    """Config for the adaptive formatter node."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    default_format: str = Field(
        default="auto",
        pattern=r"^(auto|table|markdown|json)$",
        description="Output format: auto-detect, force table, markdown, or json",
    )
    max_table_rows: int = Field(default=50, ge=1, le=1000)
    include_citations: bool = Field(default=True)
    include_metadata: bool = Field(default=False)


def _detect_format(state: GraphState, config_format: str) -> str:
    """Detect optimal format from intermediate results."""
    if config_format != "auto":
        return config_format

    results = state.get("intermediate_results", {})

    # SQL results are typically tabular
    if results.get("sql_result") or results.get("query_result"):
        return "table"

    # Check if any result looks tabular (list of dicts)
    for _key, val in results.items():
        if is_object_list(val) and val and is_object_dict(val[0]):
            return "table"

    return "markdown"


def _format_table(data: list[dict[str, object]], max_rows: int) -> str:
    """Format tabular data as markdown table."""
    if not data:
        return "_No data available._"

    truncated = data[:max_rows]
    headers = list(truncated[0].keys())

    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for row in truncated:
        cells = [str(row.get(h, ""))[:100] for h in headers]
        lines.append("| " + " | ".join(cells) + " |")

    if len(data) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(data)} rows._")

    return "\n".join(lines)


def _format_citations(sources: list[JsonDict]) -> str:
    """Format source citations."""
    if not sources:
        return ""

    lines = ["\n---\n**Sources:**"]
    for i, src in enumerate(sources[:10], 1):
        title = src.get("title", src.get("id", f"Source {i}"))
        score = src.get("score", "")
        score_str = f" (relevance: {score:.2f})" if isinstance(score, (int, float)) else ""
        lines.append(f"{i}. {title}{score_str}")

    return "\n".join(lines)


def format_output(state: GraphState) -> StateUpdate:
    """Adaptive output formatter — shapes response based on content type.

    Reads from intermediate_results/final_output and produces
    a formatted final_output for the reflect node.
    """
    config_raw = state.get("__manifest_node_config__", {})
    config = dict(config_raw) if is_object_dict(config_raw) else {}
    fmt_raw = config.get("default_format", "auto")
    fmt = fmt_raw if isinstance(fmt_raw, str) else "auto"
    max_rows_raw = config.get("max_table_rows", 50)
    max_rows = max_rows_raw if isinstance(max_rows_raw, int) else 50
    include_citations_raw = config.get("include_citations", True)
    include_citations = include_citations_raw if isinstance(include_citations_raw, bool) else True

    detected = _detect_format(state, fmt)
    results = state.get("intermediate_results", {})
    existing_output_raw = state.get("final_output", {})
    existing_output = dict(existing_output_raw) if is_object_dict(existing_output_raw) else {}

    formatted_parts: list[str] = []

    if detected == "table":
        # Find tabular data in results
        for _key, val in results.items():
            if is_object_list(val) and val and is_object_dict(val[0]):
                typed_rows = [
                    {str(column): value for column, value in row.items()}
                    for row in val
                    if is_object_dict(row)
                ]
                formatted_parts.append(_format_table(typed_rows, max_rows))
                break
        # Fallback: format sql_result
        sql = results.get("sql_result") or results.get("query_result")
        if not formatted_parts and sql:
            if is_object_list(sql):
                typed_sql = [
                    {str(column): value for column, value in row.items()}
                    for row in sql
                    if is_object_dict(row)
                ]
                formatted_parts.append(_format_table(typed_sql, max_rows))
            else:
                formatted_parts.append(str(sql))
    else:
        # Markdown: pass through existing content
        content = existing_output.get("content", "")
        if content:
            formatted_parts.append(str(content))

    # Add citations if available
    if include_citations:
        sources = results.get("sources", results.get("retrieved_documents", []))
        if is_object_list(sources) and sources:
            typed_sources: list[JsonDict] = [src for src in sources if is_json_dict(src)]
            formatted_parts.append(_format_citations(typed_sources))

    formatted_content = "\n\n".join(formatted_parts) if formatted_parts else ""

    return {
        "final_output": {
            **existing_output,
            "content": formatted_content or existing_output.get("content", ""),
            "format": detected,
        },
    }


__all__ = ["format_output", "FormatterConfig"]
