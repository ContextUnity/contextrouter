"""Stage: Build ingestion report (taxonomy/graph/shadow/export stats).

This stage is intentionally read-only with respect to core artifacts:
- Reads: taxonomy.json, knowledge_graph.pickle, shadow/*.jsonl, output/jsonl/
- Writes: output/_processing/report/ingestion_report.json

Used as guardrails to make ingestion iterations measurable and predictable.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_assets_paths
from ..graph.serialization import load_graph_secure
from ..settings import RagIngestionConfig

logger = logging.getLogger(__name__)


def _count_jsonl_lines(path: Path, *, hard_cap: int = 2_000_000) -> int:
    """Count non-empty lines in a jsonl file (bounded to avoid runaway memory)."""
    if not path.exists():
        return 0
    n = 0
    # Stream bytes to avoid loading large files.
    with open(path, "rb") as f:
        for raw in f:
            if n >= hard_cap:
                break
            if raw.strip():
                n += 1
    return n


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("report: failed to load json %s: %s", path, e)
        return None


def _taxonomy_stats(taxonomy_path: Path) -> dict[str, Any]:
    tax = _safe_json_load(taxonomy_path) or {}
    cats = tax.get("categories", {})
    canonical_map = tax.get("canonical_map", {})

    cat_keyword_counts: list[int] = []
    if isinstance(cats, dict):
        for _, cat_data in cats.items():
            if not isinstance(cat_data, dict):
                continue
            kws = cat_data.get("keywords", [])
            if isinstance(kws, list):
                cat_keyword_counts.append(len([k for k in kws if isinstance(k, str) and k.strip()]))

    cat_keyword_counts_sorted = sorted(cat_keyword_counts)
    return {
        "path": str(taxonomy_path),
        "exists": taxonomy_path.exists(),
        "categories_count": len(cats) if isinstance(cats, dict) else 0,
        "canonical_map_count": (len(canonical_map) if isinstance(canonical_map, dict) else 0),
        "total_keywords": int(tax.get("total_count") or 0),
        "category_keyword_counts": {
            "min": cat_keyword_counts_sorted[0] if cat_keyword_counts_sorted else 0,
            "p50": (
                cat_keyword_counts_sorted[len(cat_keyword_counts_sorted) // 2]
                if cat_keyword_counts_sorted
                else 0
            ),
            "p90": (
                cat_keyword_counts_sorted[int(len(cat_keyword_counts_sorted) * 0.9)]
                if cat_keyword_counts_sorted
                else 0
            ),
            "max": cat_keyword_counts_sorted[-1] if cat_keyword_counts_sorted else 0,
        },
    }


def _graph_stats(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        return {
            "path": str(graph_path),
            "exists": False,
            "nodes": 0,
            "edges": 0,
            "top_labels": [],
        }

    try:
        # Load graph with integrity verification
        g = load_graph_secure(graph_path)
    except Exception as e:
        logger.warning("report: failed to load graph %s: %s", graph_path, e)
        return {
            "path": str(graph_path),
            "exists": True,
            "load_error": str(e),
            "nodes": 0,
            "edges": 0,
            "top_labels": [],
        }

    # networkx graph-like API (duck typed)
    nodes = int(getattr(g, "number_of_nodes", lambda: 0)() or 0)
    edges = int(getattr(g, "number_of_edges", lambda: 0)() or 0)

    label_counts: Counter[str] = Counter()
    try:
        edges_fn = getattr(g, "edges", None)
        if callable(edges_fn):
            for _, _, data in edges_fn(data=True):
                rel = str((data or {}).get("relation", "")).strip()
                if rel:
                    label_counts[rel] += 1
    except Exception:
        # If g isn't a networkx graph for some reason, keep stats minimal.
        label_counts = Counter()

    top_labels = [{"label": k, "count": v} for k, v in label_counts.most_common(25)]
    return {
        "path": str(graph_path),
        "exists": True,
        "nodes": nodes,
        "edges": edges,
        "top_labels": top_labels,
    }


def _shadow_stats(shadow_dir: Path, *, types: list[str]) -> dict[str, Any]:
    per_type: dict[str, Any] = {}
    total = 0
    for t in types:
        p = shadow_dir / f"{t}.jsonl"
        cnt = _count_jsonl_lines(p)
        per_type[t] = {"path": str(p), "exists": p.exists(), "records": cnt}
        total += cnt
    return {"total_records": total, "per_type": per_type}


def _exports_stats(jsonl_dir: Path, *, types: list[str]) -> dict[str, Any]:
    per_type: dict[str, Any] = {}
    total_files = 0
    for t in types:
        d = jsonl_dir / t
        files = []
        if d.exists():
            files = sorted(
                [p for p in d.glob("*.jsonl") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        per_type[t] = {
            "dir": str(d),
            "exists": d.exists(),
            "files": len(files),
            "latest": str(files[0]) if files else "",
        }
        total_files += len(files)
    return {"total_files": total_files, "per_type": per_type}


def build_ingestion_report(
    *,
    config: RagIngestionConfig,
    types: list[str] | None = None,
) -> Path:
    """Build a single JSON report summarizing ingestion artifacts."""
    paths = get_assets_paths(config)
    processing = paths["processing"] / "report"
    processing.mkdir(parents=True, exist_ok=True)

    types_out = [
        t for t in (types or ["video", "book", "qa", "web", "knowledge"]) if isinstance(t, str)
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "taxonomy": _taxonomy_stats(paths["taxonomy"]),
            "graph": _graph_stats(paths["graph"]),
            "ontology": {
                "path": str(paths.get("ontology")),
                "exists": bool(paths.get("ontology") and paths["ontology"].exists()),
            },
            "shadow": _shadow_stats(paths["shadow"], types=types_out),
            "exports": _exports_stats(paths["jsonl"], types=types_out),
        },
    }

    out_path = processing / "ingestion_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("report: wrote %s", out_path)
    return out_path
