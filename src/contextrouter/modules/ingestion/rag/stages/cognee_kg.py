"""Stage: Cognee KG extraction (nodes + edges JSONL)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ...tools.cognee import CogneeGraphBuilder
from ..config import get_assets_paths
from ..core.utils import parallel_map, resolve_workers
from ..settings import RagIngestionConfig
from ..stages.store import read_raw_data_jsonl

logger = logging.getLogger(__name__)


def extract_cognee_kg(
    *,
    config: RagIngestionConfig,
    only_types: list[str],
    overwrite: bool = True,
    workers: int = 1,
) -> dict[str, str]:
    paths = get_assets_paths(config)
    nodes_path = paths["assets"] / "knowledge_nodes.jsonl"
    edges_path = paths["assets"] / "knowledge_edges.jsonl"

    if overwrite:
        for p in (nodes_path, edges_path):
            if p.exists():
                p.unlink()

    builder = CogneeGraphBuilder()
    if not builder.is_available():
        logger.warning("Cognee not available; KG extraction skipped")
        return {"nodes_path": str(nodes_path), "edges_path": str(edges_path)}

    def _run_one(t: str) -> tuple[list[dict], list[dict]]:
        clean_path = paths["clean_text"] / f"{t}.jsonl"
        records = read_raw_data_jsonl(clean_path)
        nodes: list[dict] = []
        edges: list[dict] = []
        for rec in records:
            entities, relations = builder.build_graph(rec.content or "")
            for ent in entities:
                name = str(ent.get("name") or ent.get("id") or "").strip()
                if not name:
                    continue
                node_id = _stable_id("concept", name)
                nodes.append(
                    {
                        "id": node_id,
                        "node_kind": "concept",
                        "content": name,
                        "struct_data": {"source_type": "knowledge"},
                    }
                )
            for rel in relations:
                src = str(rel.get("source") or "").strip()
                tgt = str(rel.get("target") or "").strip()
                relation = str(rel.get("relation") or "relates_to").strip()
                if not src or not tgt:
                    continue
                edges.append(
                    {
                        "source_id": _stable_id("concept", src),
                        "target_id": _stable_id("concept", tgt),
                        "relation": relation,
                        "weight": float(rel.get("weight") or 1.0),
                        "metadata": {},
                    }
                )
        return nodes, edges

    w = resolve_workers(config=config, workers=workers)
    results = parallel_map(only_types, _run_one, workers=w, ordered=False, swallow_exceptions=False)
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    for r in results:
        if not r:
            continue
        nodes, edges = r
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    _write_jsonl(nodes_path, all_nodes)
    _write_jsonl(edges_path, all_edges)
    logger.info("cognee_kg: nodes=%d edges=%d", len(all_nodes), len(all_edges))
    return {"nodes_path": str(nodes_path), "edges_path": str(edges_path)}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_id(prefix: str, value: str) -> str:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:16]}"


__all__ = ["extract_cognee_kg"]
