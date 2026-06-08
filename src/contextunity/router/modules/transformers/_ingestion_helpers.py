"""Shared helpers for RAG ingestion transformers resolving config from ContextUnit payloads."""

from __future__ import annotations

from collections.abc import Callable

from contextunity.brain.ingestion.rag.settings import RagIngestionConfig
from contextunity.core.sdk.payload import get_json_dict
from contextunity.core.types import ContextUnitPayload, JsonDict, is_json_dict


def resolve_rag_ingestion_config(
    payload: ContextUnitPayload,
    metadata: JsonDict,
    *,
    configured: RagIngestionConfig | None,
    default_loader: Callable[[], RagIngestionConfig],
) -> RagIngestionConfig:
    """Resolve ``RagIngestionConfig`` from payload content, metadata, or fallbacks."""
    cfg: RagIngestionConfig | None = None
    content = payload.get("content")
    if isinstance(content, RagIngestionConfig):
        cfg = content
    elif is_json_dict(content):
        cfg = RagIngestionConfig.model_validate(content)
    else:
        for key in ("ingestion_config", "config"):
            raw = metadata.get(key)
            if isinstance(raw, RagIngestionConfig):
                cfg = raw
                break
            if is_json_dict(raw):
                cfg = RagIngestionConfig.model_validate(raw)
                break
    if cfg is None and configured is not None:
        cfg = configured
    if cfg is None:
        cfg = default_loader()
    return cfg


def payload_metadata(
    unit_payload: ContextUnitPayload | None,
) -> tuple[ContextUnitPayload, JsonDict]:
    """Return typed payload and L2 metadata dict from a ContextUnit wire payload."""
    payload: ContextUnitPayload = unit_payload or {}
    metadata = get_json_dict(payload, "metadata")
    return payload, metadata


__all__ = ["payload_metadata", "resolve_rag_ingestion_config"]
