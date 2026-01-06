"""LangGraph graph definition for the RAG ingestion pipeline.

This module supports two modes:
- a default "full" ingestion graph via `build_graph()` / `compile_graph()`
- dynamic graphs built from a declarative recipe via `build_graph_from_recipe(...)`
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Literal, Sequence, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from contextrouter.core import get_core_config


class IngestionInputState(TypedDict, total=False):
    ingestion_config_path: str
    only_types: list[str]
    overwrite: bool
    workers: int

    skip_preprocess: bool
    skip_structure: bool  # replaces skip_taxonomy and skip_ontology
    skip_index: bool  # replaces skip_graph and skip_shadow
    skip_deploy: bool  # replaces skip_export, skip_deploy, and skip_report
    deploy_wait: bool


class IngestionState(IngestionInputState, total=False):
    # Filled by load step
    core_cfg: object
    ingestion_cfg: object
    assets_paths: dict[str, str]

    # Stage outputs
    clean_text_paths: dict[str, str]
    persona_path: str
    taxonomy_path: str
    ontology_path: str
    graph_path: str
    shadow_paths: dict[str, str]
    export_paths: dict[str, str]
    deploy_result: dict[str, object]
    report_path: str

    # Stage statuses
    stage_preprocess: str
    stage_persona: str
    stage_structure: str  # merged taxonomy+ontology
    stage_index: str  # merged graph+shadow
    stage_deploy: str  # merged export+deploy+report


class IngestionOutputState(TypedDict, total=False):
    assets_paths: dict[str, str]
    clean_text_paths: dict[str, str]
    taxonomy_path: str
    ontology_path: str
    graph_path: str
    shadow_paths: dict[str, str]
    export_paths: dict[str, str]
    deploy_result: dict[str, object]
    report_path: str


StageName = Literal[
    "load",
    "preprocess",
    "persona",
    "structure",  # merged taxonomy+ontology
    "index",  # merged graph+shadow
    "deploy",  # merged export+deploy+report
]


class IngestionRecipe(TypedDict, total=False):
    """Declarative ingestion graph recipe.

    This is safe to accept from transports (API/CLI) because it is NOT code: it is a
    constrained selection of allowed stages.

    - `stages`: subset of stages to include in the graph.
      If omitted/empty, the full default pipeline is used.
    - `allow_unsafe`: if True, allows selecting stages without their required predecessors.
      Default is False (safe).
    """

    stages: list[StageName]
    allow_unsafe: bool


class IngestionJobSpec(TypedDict, total=False):
    """API-friendly ingestion job specification (JSON-safe).

    - `recipe`: graph wiring (which stages to include).
    - `input`: graph input parameters (a subset of IngestionInputState).

    This is safe to store as a job payload and replay later.
    """

    recipe: IngestionRecipe
    input: IngestionInputState


_DEFAULT_STAGE_ORDER: list[StageName] = [
    "load",
    "preprocess",
    "persona",
    "structure",  # merged taxonomy+ontology
    "index",  # merged graph+shadow
    "deploy",  # merged export+deploy+report
]


_REQUIRES: dict[StageName, set[StageName]] = {
    # Minimal "safe" constraints (best-effort).
    # We keep this conservative: if you're sure you have artifacts already,
    # set recipe.allow_unsafe=True.
    "load": set(),
    "preprocess": {"load"},
    "persona": {"load"},
    "structure": {"load", "preprocess"},  # requires preprocess (for taxonomy)
    "index": {"load", "preprocess"},  # requires preprocess (for graph/shadow)
    "deploy": {"load"},  # can run independently (has export internally)
}


def _normalize_stages(stages: Sequence[str] | None) -> list[StageName]:
    if not stages:
        return list(_DEFAULT_STAGE_ORDER)
    raw: list[str] = [s.strip() for s in stages if isinstance(s, str) and s.strip()]
    allowed = set(_DEFAULT_STAGE_ORDER)
    unknown = sorted({s for s in raw if s not in allowed})
    if unknown:
        raise ValueError(f"Unknown ingestion stages: {unknown}. Allowed: {sorted(allowed)}")
    wanted = set(cast(set[StageName], set(raw)))
    # Always include load unless explicitly excluded by allow_unsafe (handled elsewhere).
    wanted.add("load")
    return [s for s in _DEFAULT_STAGE_ORDER if s in wanted]


def _validate_dependencies(stages: Sequence[StageName], *, allow_unsafe: bool) -> None:
    if allow_unsafe:
        return
    present = set(stages)
    missing: dict[str, list[str]] = {}
    for s in stages:
        req = _REQUIRES.get(s, set())
        bad = sorted({r for r in req if r not in present})
        if bad:
            missing[str(s)] = bad
    if missing:
        raise ValueError(
            "IngestionRecipe violates stage dependencies (set allow_unsafe=True to bypass): "
            + ", ".join([f"{k} missing {v}" for k, v in missing.items()])
        )


def resolve_recipe_stages(recipe: IngestionRecipe | None) -> list[StageName]:
    """Resolve the effective stage list for a recipe (safe, canonical order)."""
    if not isinstance(recipe, dict):
        return list(_DEFAULT_STAGE_ORDER)
    stages = _normalize_stages(recipe.get("stages"))
    allow_unsafe = bool(recipe.get("allow_unsafe", False))
    _validate_dependencies(stages, allow_unsafe=allow_unsafe)
    return stages


def recipe_cache_key(recipe: IngestionRecipe) -> str:
    """Stable cache key for compiled graphs from a recipe."""
    stages = resolve_recipe_stages(recipe)
    allow_unsafe = bool(recipe.get("allow_unsafe", False))
    payload = {"stages": list(stages), "allow_unsafe": allow_unsafe}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()  # noqa: S324


_compiled_recipe_graphs: dict[str, object] = {}


def build_graph_from_recipe(recipe: IngestionRecipe) -> StateGraph:
    """Build an ingestion graph from a declarative recipe.

    NOTE: This does NOT consult `router.override_path`.
    If you want to fully override ingestion wiring, call your override builder directly.
    """
    stages = resolve_recipe_stages(recipe)

    from contextrouter.cortex.steps.rag_ingestion import (
        load_ingestion_config,
        stage_deploy,  # merged export+deploy+report
        stage_index,  # merged graph+shadow
        stage_persona,
        stage_preprocess,
        stage_structure,  # merged taxonomy+ontology
    )

    nodes: dict[StageName, Callable[[dict[str, object]], dict[str, object]]] = {
        "load": load_ingestion_config,
        "preprocess": stage_preprocess,
        "persona": stage_persona,
        "structure": stage_structure,  # merged taxonomy+ontology
        "index": stage_index,  # merged graph+shadow
        "deploy": stage_deploy,  # merged export+deploy+report
    }

    workflow = StateGraph(IngestionState, input=IngestionInputState, output=IngestionOutputState)
    for s in stages:
        workflow.add_node(s, nodes[s])

    workflow.add_edge(START, stages[0])
    for left, right in zip(stages, stages[1:], strict=False):
        workflow.add_edge(left, right)
    workflow.add_edge(stages[-1], END)
    return workflow


def compile_graph_from_recipe(recipe: IngestionRecipe) -> object:
    """Compile and memoize a graph for a recipe (process-local cache)."""
    key = recipe_cache_key(recipe)
    cached = _compiled_recipe_graphs.get(key)
    if cached is not None:
        return cached
    compiled = build_graph_from_recipe(recipe).compile()
    _compiled_recipe_graphs[key] = compiled
    return compiled


def build_graph() -> StateGraph:
    # Host override (power-user wiring)
    override_path = get_core_config().router.override_path
    if override_path:
        # Let override handle ingestion graph completely (same convention as retrieval graph)
        import importlib

        raw = override_path.strip()
        if ":" in raw:
            mod_name, attr = raw.split(":", 1)
        else:
            mod_name, attr = raw.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        if hasattr(obj, "build_graph"):
            return obj.build_graph()  # type: ignore[no-any-return]
        if callable(obj):
            return obj()  # type: ignore[no-any-return]
        raise TypeError(f"router.override_path object is not callable: {override_path}")

    from contextrouter.cortex.steps.rag_ingestion import (
        load_ingestion_config,
        stage_deploy,  # merged export+deploy+report
        stage_index,  # merged graph+shadow
        stage_persona,
        stage_preprocess,
        stage_structure,  # merged taxonomy+ontology
    )

    workflow = StateGraph(IngestionState, input=IngestionInputState, output=IngestionOutputState)

    workflow.add_node("load", load_ingestion_config)
    workflow.add_node("preprocess", stage_preprocess)
    workflow.add_node("persona", stage_persona)
    workflow.add_node("structure", stage_structure)  # merged taxonomy+ontology
    workflow.add_node("index", stage_index)  # merged graph+shadow
    workflow.add_node("deploy", stage_deploy)  # merged export+deploy+report

    workflow.add_edge(START, "load")
    workflow.add_edge("load", "preprocess")
    workflow.add_edge("preprocess", "persona")
    workflow.add_edge("persona", "structure")
    workflow.add_edge("structure", "index")
    workflow.add_edge("index", "deploy")
    workflow.add_edge("deploy", END)

    return workflow


_compiled_graph: object | None = None


def compile_graph() -> object:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return cast(object, _compiled_graph)


def reset_graph() -> None:
    global _compiled_graph
    _compiled_graph = None
    _compiled_recipe_graphs.clear()


__all__ = [
    "IngestionInputState",
    "IngestionState",
    "IngestionOutputState",
    "IngestionRecipe",
    "IngestionJobSpec",
    "StageName",
    "build_graph",
    "build_graph_from_recipe",
    "compile_graph_from_recipe",
    "recipe_cache_key",
    "resolve_recipe_stages",
    "compile_graph",
    "reset_graph",
]
