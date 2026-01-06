"""Record creation and JSONL utilities."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, TypedDict

from contextrouter.core.types import StructData, coerce_struct_data


class VertexImportContent(TypedDict):
    mimeType: str
    rawBytes: str


class VertexImportRecord(TypedDict):
    id: str
    content: VertexImportContent
    structData: StructData


def generate_id(*parts: str) -> str:
    """Generate a stable MD5 hash ID from string parts."""
    combined = "_".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def slugify(value: str, max_length: int = 50) -> str:
    """Convert string to URL-safe slug."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    if max_length and len(value) > max_length:
        value = value[:max_length].rstrip("_")
    return value or "item"


def create_record(
    record_id: str,
    content: str,
    source_type: str,
    title: str,
    **metadata,
) -> VertexImportRecord:
    """Create a JSONL record for Vertex AI Search."""
    struct_data: StructData = {"source_type": source_type, "title": title}
    for k, v in metadata.items():
        if isinstance(k, str):
            struct_data[k] = coerce_struct_data(v)

    return {
        "id": record_id,
        "content": {
            "mimeType": "text/plain",
            "rawBytes": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        },
        "structData": struct_data,
    }


def write_jsonl(records: Iterable[VertexImportRecord], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
