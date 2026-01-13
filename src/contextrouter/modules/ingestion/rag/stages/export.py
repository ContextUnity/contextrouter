"""Stage 5: ShadowRecords -> per-type Vertex import JSONL."""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime

from contextrouter.core.types import StructData

from ..config import get_assets_paths
from ..core.types import ShadowRecord
from ..core.utils import parallel_map, resolve_workers
from ..settings import RagIngestionConfig
from .store import read_shadow_records_jsonl

logger = logging.getLogger(__name__)


def export_jsonl_per_type(
    *,
    config: RagIngestionConfig,
    only_types: list[str],
    overwrite: bool = True,
    workers: int = 1,
) -> dict[str, str]:
    paths = get_assets_paths(config)
    db_name = config.upload.db_name
    include_date = config.upload.include_date

    date_str = datetime.now().strftime("%Y%m%d")

    out: dict[str, str] = {}

    def _run_one(t: str) -> tuple[str, str]:
        t0 = time.perf_counter()
        shadow_path = paths["shadow"] / f"{t}.jsonl"
        records = read_shadow_records_jsonl(shadow_path)
        if not records:
            logger.warning("export: no shadow records for type=%s at %s", t, shadow_path)
            return (t, "")

        type_dir = paths["jsonl"] / t
        type_dir.mkdir(parents=True, exist_ok=True)

        if include_date:
            filename = f"import_{date_str}_{db_name}_{t}.jsonl"
        else:
            filename = f"import_{db_name}_{t}.jsonl"

        out_path = type_dir / filename

        mode = "w" if overwrite else "a"
        with open(out_path, mode, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(_to_vertex_record(r), ensure_ascii=False) + "\n")

        logger.warning(
            "export: wrote %d records for type=%s -> %s (%.1fs)",
            len(records),
            t,
            out_path,
            time.perf_counter() - t0,
        )
        return (t, str(out_path))

    w = resolve_workers(config=config, workers=workers)
    results = parallel_map(only_types, _run_one, workers=w, ordered=False, swallow_exceptions=False)
    for r in results:
        if not r:
            continue
        tt, out_path = r
        if out_path:
            out[tt] = out_path

    return out


def _to_vertex_record(record: ShadowRecord) -> StructData:
    struct_data: StructData = dict(record.struct_data or {})
    if record.source_type:
        struct_data["source_type"] = record.source_type
    if record.title:
        struct_data["title"] = record.title
    if record.citation_label:
        struct_data["citation_label"] = record.citation_label

    return {
        "id": record.id,
        "content": {
            "mimeType": "text/plain",
            "rawBytes": base64.b64encode(record.input_text.encode("utf-8")).decode("utf-8"),
        },
        "structData": struct_data,
    }
