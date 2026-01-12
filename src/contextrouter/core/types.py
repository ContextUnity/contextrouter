"""Shared, vendor-neutral types used across contextrouter.

Keep these types *generic* and reusable across:
- brain (LangGraph agent)
- ingestion (data preparation)
- integrations (format adapters)
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

# ---- StructData typing -------------------------------------------------------
#
# We intentionally avoid `Any` in most of the codebase. When we deal with data
# that must be JSON-serializable (e.g., JSONL ingestion artifacts), we use these
# recursive aliases.
type StructDataPrimitive = str | int | float | bool | None
type StructDataValue = StructDataPrimitive | list["StructDataValue"] | dict[str, "StructDataValue"]
type StructData = dict[str, StructDataValue]


def coerce_struct_data(value: object) -> StructDataValue:
    """Best-effort conversion into JSON-serializable StructDataValue.

    This is intentionally conservative and used at integration boundaries where
    external SDKs return loosely-typed Python objects.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out: StructData = {}
        for k, v in value.items():
            out[str(k)] = coerce_struct_data(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [coerce_struct_data(v) for v in value]
    # Fallback: stringify unknown objects (keeps JSON serializable)
    return str(value)


SourceType: TypeAlias = str


class TextQuery(TypedDict):
    """A plain text query."""

    kind: Literal["text"]
    text: str


class SqlQuery(TypedDict, total=False):
    """A structured SQL query.

    This is *transported* through contextrouter but executed only by providers that
    understand it (e.g., a Postgres analytics provider).
    """

    kind: Literal["sql"]
    sql: str
    dialect: NotRequired[str]
    params: NotRequired[StructData]


class QueryPayload(TypedDict, total=False):
    """Generic structured query payload for custom retrievers/providers."""

    kind: str
    data: NotRequired[StructData]


Query: TypeAlias = TextQuery | SqlQuery | QueryPayload
QueryLike: TypeAlias = str | Query


def normalize_query(query: object) -> tuple[str, dict[str, Any] | None]:
    """Normalize QueryLike into `(query_text, extra_filters)` without breaking IRead/IWrite.

    Compatibility rule:
    - Providers still receive `query: str` (per IRead.read signature).
    - Structured information is passed via `filters` (extra_filters) for providers that can use it.
    """

    if isinstance(query, str):
        return query, None

    # Runtime safety: callers should pass QueryLike, but integrations may pass arbitrary objects.
    # Keep this defensive without making pyright consider branches unreachable.
    if not isinstance(query, dict):
        return str(query), {"query_kind": "unknown"}

    kind = str(query.get("kind") or "text").strip() or "text"

    if kind == "text":
        text = query.get("text")
        return (text if isinstance(text, str) else str(text)), {"query_kind": "text"}

    if kind == "sql":
        sql = query.get("sql")
        extra: dict[str, Any] = {"query_kind": "sql"}
        extra["sql"] = sql if isinstance(sql, str) else str(sql)
        if (dialect := query.get("dialect")) is not None:
            extra["sql_dialect"] = dialect
        if (params := query.get("params")) is not None:
            extra["sql_params"] = params
        # For compatibility: pass the SQL as the `query` string too.
        return extra["sql"], extra

    # Unknown structured kind: stringify and attach payload into filters.
    extra = {"query_kind": kind}
    if "data" in query:
        extra["query_data"] = query.get("data")
    return kind, extra


class UserCtx(TypedDict, total=False):
    """Authenticated user context passed from host apps (api/telegram) to the brain."""

    user_id: str
    role: str
    permissions: list[str]
    tenant_id: str | None


__all__ = [
    "StructDataPrimitive",
    "StructDataValue",
    "StructData",
    "coerce_struct_data",
    "SourceType",
    "TextQuery",
    "SqlQuery",
    "QueryPayload",
    "Query",
    "QueryLike",
    "normalize_query",
    "UserCtx",
]
