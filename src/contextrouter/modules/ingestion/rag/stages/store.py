"""Persistence helpers for staged ingestion artifacts (JSONL)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from contextrouter.core.types import StructData, StructDataValue

from ..core import RawData, ShadowRecord

logger = logging.getLogger(__name__)


def write_raw_data_jsonl(items: list[RawData], path: Path, *, overwrite: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with open(path, mode, encoding="utf-8") as f:
        for item in items:
            payload = {
                "content": item.content,
                "source_type": item.source_type,
                "metadata": item.metadata,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(items)


def read_raw_data_jsonl(path: Path) -> list[RawData]:
    out: list[RawData] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj: StructDataValue = json.loads(line)
        except Exception as e:
            logger.debug("Failed to parse JSON line: %s", e)
            continue
        if not isinstance(obj, dict):
            continue
        obj_dict: StructData = {k: v for k, v in obj.items() if isinstance(k, str)}
        content = obj_dict.get("content")
        source_type = obj_dict.get("source_type")
        metadata = obj_dict.get("metadata") or {}
        if isinstance(content, str) and isinstance(source_type, str) and isinstance(metadata, dict):
            out.append(RawData(content=content, source_type=source_type, metadata=metadata))
    return out


def write_shadow_records_jsonl(
    records: list[ShadowRecord],
    path: Path,
    *,
    overwrite: bool = True,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            payload = {
                "id": r.id,
                "input_text": r.input_text,
                "struct_data": r.struct_data,
                "citation_label": r.citation_label,
                "title": r.title,
                "source_type": r.source_type,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(records)


def read_shadow_records_jsonl(path: Path) -> list[ShadowRecord]:
    out: list[ShadowRecord] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj: StructDataValue = json.loads(line)
        except Exception as e:
            logger.debug("Failed to parse JSON line: %s", e)
            continue
        if not isinstance(obj, dict):
            continue
        obj_dict: StructData = {k: v for k, v in obj.items() if isinstance(k, str)}
        rid = obj_dict.get("id")
        input_text = obj_dict.get("input_text")
        struct_data = obj_dict.get("struct_data")
        if (
            not isinstance(rid, str)
            or not isinstance(input_text, str)
            or not isinstance(struct_data, dict)
        ):
            continue
        out.append(
            ShadowRecord(
                id=rid,
                input_text=input_text,
                struct_data=struct_data,
                citation_label=(
                    obj_dict.get("citation_label")
                    if isinstance(obj_dict.get("citation_label"), str)
                    else None
                ),
                title=(obj_dict.get("title") if isinstance(obj_dict.get("title"), str) else None),
                source_type=(
                    obj_dict.get("source_type")
                    if isinstance(obj_dict.get("source_type"), str)
                    else None
                ),
            )
        )
    return out
