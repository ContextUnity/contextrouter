"""Shared, vendor-neutral types used across contextunity.router.

Keep these types *generic* and reusable across:
- brain (LangGraph agent)
- ingestion (data preparation)
- integrations (format adapters)
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

# ---- StructData typing -------------------------------------------------------
# Re-exported from core — canonical definition lives in core.sdk.types.
from contextunity.core.sdk.types import (
    StructData,
    StructDataPrimitive,
    StructDataValue,
    coerce_struct_data,
)
from contextunity.core.types import JsonDict, is_object_dict

SourceType: TypeAlias = str


class TextQuery(TypedDict):
    """A plain text query."""

    kind: Literal["text"]
    text: str


class SqlQuery(TypedDict, total=False):
    """A structured SQL query.

    This is *transported* through contextunity.router but executed only by providers that
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


def normalize_query(query: object) -> tuple[str, JsonDict | None]:
    """Normalize QueryLike into `(query_text, extra_filters)` without breaking IRead/IWrite.

    Compatibility rule:
    - Providers still receive `query: str` (per IRead.read signature).
    - Structured information is passed via `filters` (extra_filters) for providers that can use it.
    """

    if isinstance(query, str):
        return query, None

    # Runtime safety: callers should pass QueryLike, but integrations may pass arbitrary objects.
    if not is_object_dict(query):
        return str(query), {"query_kind": "unknown"}

    payload: dict[str, object] = query
    kind_obj = payload.get("kind")
    if isinstance(kind_obj, str):
        kind = kind_obj.strip() or "text"
    else:
        kind = "text"

    if kind == "text":
        text_obj = payload.get("text")
        text = text_obj if isinstance(text_obj, str) else str(text_obj)
        return text, {"query_kind": "text"}

    if kind == "sql":
        sql_obj = payload.get("sql")
        sql_text = sql_obj if isinstance(sql_obj, str) else str(sql_obj)
        sql_extra: JsonDict = {"query_kind": "sql", "sql": sql_text}
        dialect_obj = payload.get("dialect")
        if dialect_obj is not None:
            sql_extra["sql_dialect"] = (
                dialect_obj if isinstance(dialect_obj, str) else str(dialect_obj)
            )
        params_obj = payload.get("params")
        if params_obj is not None:
            sql_extra["sql_params"] = coerce_struct_data(params_obj)
        return sql_text, sql_extra

    kind_extra: JsonDict = {"query_kind": kind}
    data_obj = payload.get("data")
    if data_obj is not None:
        kind_extra["query_data"] = coerce_struct_data(data_obj)
    return kind, kind_extra


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
