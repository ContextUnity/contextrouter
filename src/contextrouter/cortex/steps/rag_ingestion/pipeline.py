"""Ingestion graph: stage execution steps.

This wraps existing ingestion capability functions from `modules.ingestion.rag`.
We keep these as thin orchestrators (no business logic duplication).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contextrouter.modules.ingestion.rag import (
    build_graph_from_clean_text,
    build_ingestion_report,
    build_ontology_from_taxonomy,
    build_persona,
    build_shadow_records,
    build_taxonomy_from_clean_text,
    deploy_jsonl_files,
    export_jsonl_per_type,
    preprocess_to_clean_text,
)


def _paths(state: dict[str, Any]) -> dict[str, Path]:
    raw = state.get("assets_paths") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Path] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = Path(v)
    return out


def stage_preprocess(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("skip_preprocess") is True:
        return {"stage_preprocess": "skipped"}
    cfg = state["ingestion_cfg"]
    core_cfg = state["core_cfg"]
    only_types = state["only_types"]
    overwrite = bool(state.get("overwrite", True))
    workers = int(state.get("workers", 0) or 0)

    out = preprocess_to_clean_text(
        core_cfg=core_cfg,
        config=cfg,
        only_types=list(only_types),
        overwrite=overwrite,
        workers=workers if workers > 0 else 1,
    )
    return {"clean_text_paths": {k: str(v) for k, v in out.items()}, "stage_preprocess": "ok"}


def stage_persona(state: dict[str, Any]) -> dict[str, Any]:
    # Persona stage is optional; controlled by ingestion config.
    cfg = state["ingestion_cfg"]
    core_cfg = state["core_cfg"]
    out = build_persona(config=cfg, core_cfg=core_cfg)
    return {"persona_path": str(out) if out is not None else "", "stage_persona": "ok"}


def stage_structure(state: dict[str, Any]) -> dict[str, Any]:
    """Merged taxonomy + ontology stage."""
    if state.get("skip_taxonomy") is True or state.get("skip_ontology") is True:
        return {"stage_structure": "skipped"}

    cfg = state["ingestion_cfg"]
    core_cfg = state["core_cfg"]
    overwrite = bool(state.get("overwrite", True))
    workers = int(state.get("workers", 0) or 0)

    # Build taxonomy
    taxonomy_path = build_taxonomy_from_clean_text(
        config=cfg, core_cfg=core_cfg, force=overwrite, workers=workers
    )

    # Build ontology from taxonomy
    ontology_path = build_ontology_from_taxonomy(config=cfg, overwrite=overwrite)

    return {
        "taxonomy_path": str(taxonomy_path) if taxonomy_path is not None else "",
        "ontology_path": str(ontology_path) if ontology_path is not None else "",
        "stage_structure": "ok",
    }


def stage_index(state: dict[str, Any]) -> dict[str, Any]:
    """Index stage: graph building + shadow records creation."""
    cfg = state["ingestion_cfg"]
    core_cfg = state["core_cfg"]
    only_types = state["only_types"]
    overwrite = bool(state.get("overwrite", True))
    workers = int(state.get("workers", 0) or 0)

    result = {"stage_index": "ok"}

    # Build graph if not skipped
    if state.get("skip_graph") is not True:
        graph_path = build_graph_from_clean_text(
            config=cfg, core_cfg=core_cfg, workers=workers, overwrite=overwrite
        )
        result["graph_path"] = str(graph_path) if graph_path is not None else ""
    else:
        result["graph_path"] = state.get("graph_path", "")  # Keep existing

    # Build shadow records if not skipped
    if state.get("skip_shadow") is not True:
        shadow_paths = build_shadow_records(
            config=cfg,
            core_cfg=core_cfg,
            only_types=list(only_types),
            overwrite=overwrite,
            workers=workers if workers > 0 else 1,
        )
        result["shadow_paths"] = shadow_paths
    else:
        result["shadow_paths"] = state.get("shadow_paths", {})  # Keep existing

    return result


def stage_deploy(state: dict[str, Any]) -> dict[str, Any]:
    """Deploy stage: export + deploy + report."""
    cfg = state["ingestion_cfg"]
    only_types = state["only_types"]
    overwrite = bool(state.get("overwrite", True))
    workers = int(state.get("workers", 0) or 0)
    wait = bool(state.get("deploy_wait", False))

    result = {"stage_deploy": "ok"}

    # Export JSONL files if not skipped
    if state.get("skip_export") is not True:
        export_paths = export_jsonl_per_type(
            config=cfg,
            only_types=list(only_types),
            overwrite=overwrite,
            workers=workers if workers > 0 else 1,
        )
        result["export_paths"] = export_paths
    else:
        export_paths = state.get("export_paths", {})
        result["export_paths"] = export_paths

    # Deploy to search index if not skipped
    if state.get("skip_deploy") is not True and export_paths:
        deploy_result = deploy_jsonl_files(jsonl_paths_by_type=export_paths, config=cfg, wait=wait)
        # Make result JSON-ish
        deploy_result = {
            str(k): getattr(v, "model_dump", lambda: {"success": getattr(v, "success", False)})()
            for k, v in deploy_result.items()
        }
        result["deploy_result"] = deploy_result
    else:
        result["deploy_result"] = state.get("deploy_result", {})

    # Generate report if not skipped
    if state.get("skip_report") is not True:
        report_path = build_ingestion_report(config=cfg, only_types=list(only_types))
        result["report_path"] = str(report_path) if report_path is not None else ""
    else:
        result["report_path"] = state.get("report_path", "")

    return result


__all__ = [
    "stage_preprocess",
    "stage_persona",
    "stage_structure",  # merged taxonomy+ontology
    "stage_index",  # merged graph+shadow
    "stage_deploy",  # merged export+deploy+report
]
