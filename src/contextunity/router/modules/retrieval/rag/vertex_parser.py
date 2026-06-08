"""Vertex AI Search result parser — RAG domain logic.

Converts raw Discovery Engine protobuf results into ``RetrievedDoc``
with source_type heuristics and field enrichment.

Extracted from ``providers/storage/vertex_search.py`` to enforce
the storage ↔ RAG boundary.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonValue, is_json_dict, is_object_dict, is_object_list
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from contextunity.router.core.types import coerce_struct_data

from .models import RetrievedDoc

logger = get_contextunit_logger(__name__)


@runtime_checkable
class SearchResultEnvelope(Protocol):
    @property
    def document(self) -> object: ...

    @property
    def relevance_score(self) -> object: ...


@runtime_checkable
class SearchDocumentEnvelope(Protocol):
    @property
    def derived_struct_data(self) -> object: ...

    @property
    def struct_data(self) -> object: ...


def _proto_to_dict(proto_obj: object) -> JsonValue:
    """Convert protobuf object to JSON-like tree."""
    if proto_obj is None:
        return None

    if isinstance(proto_obj, Message):
        try:
            return MessageToDict(proto_obj, preserving_proto_field_name=True)
        except Exception as e:
            logger.debug("MessageToDict failed, trying fallback conversion: %s", e)
    if is_object_dict(proto_obj):
        return {str(k): _proto_to_dict(v) for k, v in proto_obj.items()}
    if isinstance(proto_obj, (str, int, float, bool)):
        return proto_obj
    return str(proto_obj)


def _string_dict(data: object) -> dict[str, object]:
    if not is_object_dict(data):
        return {}
    return {str(k): v for k, v in data.items()}


def parse_search_result(result: object) -> RetrievedDoc | None:
    """Parse Discovery Engine search result to RetrievedDoc."""
    try:
        if not isinstance(result, SearchResultEnvelope):
            return None
        doc = result.document
        if doc is None:
            return None

        derived_raw = doc.derived_struct_data if isinstance(doc, SearchDocumentEnvelope) else None
        struct_raw = doc.struct_data if isinstance(doc, SearchDocumentEnvelope) else None
        derived_data = _string_dict(_proto_to_dict(derived_raw))
        struct_data = _string_dict(_proto_to_dict(struct_raw))
        doc_data: dict[str, object] = {**struct_data, **derived_data}

        def _get_str(k: str) -> str:
            v = doc_data.get(k)
            return str(v) if v is not None else ""

        def _get_int(k: str) -> int | None:
            v = doc_data.get(k)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.isdigit():
                return int(v)
            return None

        def _get_list(k: str) -> list[str]:
            v = doc_data.get(k)
            if is_object_list(v):
                return [str(x) for x in v if x is not None]
            return []

        content = ""
        ea = doc_data.get("extractive_answers")
        if is_object_list(ea) and ea and is_object_dict(ea[0]):
            content_raw = ea[0].get("content")
            content = content_raw if isinstance(content_raw, str) else ""
        else:
            snippets = doc_data.get("snippets")
            if is_object_list(snippets) and snippets and is_object_dict(snippets[0]):
                snippet_raw = snippets[0].get("snippet")
                content = snippet_raw if isinstance(snippet_raw, str) else ""

        if not content:
            c = doc_data.get("content")
            t = doc_data.get("text")
            if isinstance(c, str) and c:
                content = c
            elif isinstance(t, str) and t:
                content = t

        st_raw = str(doc_data.get("source_type", ""))
        source_type: Literal["book", "video", "qa", "web", "knowledge", "unknown"] = "unknown"
        for val in ("book", "video", "qa", "web", "knowledge"):
            if st_raw == val:
                source_type = val
                break

        if source_type == "unknown":
            if (
                doc_data.get("book_title")
                or doc_data.get("chapter_title")
                or doc_data.get("page_start") is not None
            ):
                source_type = "book"
            elif (
                doc_data.get("video_id") or doc_data.get("video_url") or doc_data.get("video_name")
            ):
                source_type = "video"
            elif (
                doc_data.get("session_title") or doc_data.get("question") or doc_data.get("answer")
            ):
                source_type = "qa"

        relevance_raw = result.relevance_score
        relevance = (
            float(relevance_raw or 0.0) if isinstance(relevance_raw, (int, float, str)) else 0.0
        )

        parsed = RetrievedDoc(
            content=content,
            source_type=source_type,
            relevance=relevance,
        )

        if source_type == "book":
            parsed.book_title = _get_str("book_title")
            parsed.chapter = _get_str("chapter_title") or _get_str("chapter")
            parsed.chapter_number = _get_int("chapter_number")
            parsed.page_start = _get_int("page_start")
            parsed.page_end = _get_int("page_end")
            parsed.keywords = _get_list("keywords")
            quote = _get_str("quote")
            if quote.strip():
                parsed.quote = quote
                parsed.content = quote
        elif source_type == "video":
            parsed.video_id = _get_str("video_id")
            parsed.video_url = _get_str("video_url")
            parsed.title = _get_str("video_name") or _get_str("title")
            parsed.timestamp = _get_str("timestamp")
            parsed.timestamp_seconds = _get_int("timestamp_seconds")
            parsed.keywords = _get_list("keywords")
            quote = _get_str("quote")
            if quote.strip():
                parsed.quote = quote
                parsed.content = quote
        elif source_type == "qa":
            parsed.session_title = _get_str("session_title")
            parsed.question = _get_str("question")
            parsed.answer = _get_str("answer")
            parsed.keywords = _get_list("keywords")

        metadata_val = coerce_struct_data(doc_data)
        parsed.metadata = metadata_val if is_json_dict(metadata_val) else {}
        return parsed
    except Exception:
        return None


__all__ = ["parse_search_result"]
