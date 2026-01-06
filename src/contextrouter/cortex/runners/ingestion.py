"""Ingestion graph runner (async).

This is the host-facing entrypoint for running ingestion as a background job and
streaming progress events.

It intentionally emits **normalized progress events** (not raw LangGraph events)
so transports (API, CLI, workers) can remain simple and stable.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator, Literal, NotRequired, TypedDict, cast

from contextrouter.core import TokenBuilder, UserCtx, get_core_config
from contextrouter.cortex.graphs import rag_ingestion
from contextrouter.cortex.graphs.rag_ingestion import (
    IngestionJobSpec,
    IngestionRecipe,
    StageName,
)

logger = logging.getLogger(__name__)


class IngestionProgressEvent(TypedDict, total=False):
    """Normalized ingestion progress event."""

    type: Literal[
        "run_started",
        "stage_started",
        "stage_finished",
        "stage_failed",
        "run_finished",
        "run_failed",
    ]
    run_id: str
    ts: float
    graph: str
    stage: NotRequired[StageName]
    status: NotRequired[str]
    error: NotRequired[str]
    output_keys: NotRequired[list[str]]


def _now() -> float:
    return time.time()


def _mk_input_state(
    *,
    ingestion_config_path: str = "",
    only_types: list[str] | None = None,
    overwrite: bool = True,
    workers: int = 1,
    # switches
    skip_preprocess: bool = False,
    skip_taxonomy: bool = False,
    skip_ontology: bool = False,
    skip_graph: bool = False,
    skip_shadow: bool = False,
    skip_export: bool = False,
    skip_deploy: bool = False,
    skip_report: bool = False,
    deploy_wait: bool = False,
) -> dict[str, object]:
    state: dict[str, object] = {
        "ingestion_config_path": ingestion_config_path,
        "only_types": list(only_types or []),
        "overwrite": bool(overwrite),
        "workers": int(workers),
        "skip_preprocess": bool(skip_preprocess),
        "skip_taxonomy": bool(skip_taxonomy),
        "skip_ontology": bool(skip_ontology),
        "skip_graph": bool(skip_graph),
        "skip_shadow": bool(skip_shadow),
        "skip_export": bool(skip_export),
        "skip_deploy": bool(skip_deploy),
        "skip_report": bool(skip_report),
        "deploy_wait": bool(deploy_wait),
    }
    return state


def _input_state_from_spec(
    spec: IngestionJobSpec,
) -> tuple[IngestionRecipe | None, dict[str, object]]:
    """Extract (recipe, input_state) from a job spec.

    The spec is JSON-shaped, so we defensively normalize types.
    """
    recipe: IngestionRecipe | None = None
    raw_recipe = spec.get("recipe") if isinstance(spec, dict) else None
    if isinstance(raw_recipe, dict):
        recipe = cast(IngestionRecipe, raw_recipe)

    raw_input = spec.get("input") if isinstance(spec, dict) else None
    input_state: dict[str, object] = {}
    if isinstance(raw_input, dict):
        input_state = {k: v for k, v in raw_input.items() if isinstance(k, str)}
    return recipe, input_state


def _extract_output_keys(event: dict[str, object]) -> list[str]:
    data = event.get("data")
    if not isinstance(data, dict):
        return []
    output = data.get("output")
    if isinstance(output, dict):
        return [k for k in output.keys() if isinstance(k, str)]
    return []


def _extract_stage_status(event: dict[str, object], stage: str) -> str | None:
    """Best-effort extract of stage status from node output."""
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    output = data.get("output")
    if not isinstance(output, dict):
        return None
    key = f"stage_{stage}"
    v = output.get(key)
    if isinstance(v, str) and v.strip():
        return v
    return None


async def invoke_ingestion(
    *,
    spec: IngestionJobSpec | None = None,
    recipe: IngestionRecipe | None = None,
    ingestion_config_path: str = "",
    only_types: list[str] | None = None,
    overwrite: bool = True,
    workers: int = 1,
    skip_preprocess: bool = False,
    skip_taxonomy: bool = False,
    skip_ontology: bool = False,
    skip_graph: bool = False,
    skip_shadow: bool = False,
    skip_export: bool = False,
    skip_deploy: bool = False,
    skip_report: bool = False,
    deploy_wait: bool = False,
    user_ctx: UserCtx | None = None,
) -> dict[str, object]:
    """Run ingestion to completion (non-streaming)."""
    if isinstance(spec, dict):
        recipe_from_spec, input_state = _input_state_from_spec(spec)
        if recipe_from_spec is not None:
            recipe = recipe_from_spec
    else:
        input_state = _mk_input_state(
            ingestion_config_path=ingestion_config_path,
            only_types=only_types,
            overwrite=overwrite,
            workers=workers,
            skip_preprocess=skip_preprocess,
            skip_taxonomy=skip_taxonomy,
            skip_ontology=skip_ontology,
            skip_graph=skip_graph,
            skip_shadow=skip_shadow,
            skip_export=skip_export,
            skip_deploy=skip_deploy,
            skip_report=skip_report,
            deploy_wait=deploy_wait,
        )

    graph = (
        rag_ingestion.compile_graph_from_recipe(recipe)
        if isinstance(recipe, dict)
        else rag_ingestion.compile_graph()
    )

    core_cfg = get_core_config()
    if core_cfg.security.enabled:
        builder = TokenBuilder(enabled=True, private_key_path=core_cfg.security.private_key_path)
        token = builder.mint_root(
            user_ctx=user_ctx or {},
            permissions=(
                core_cfg.security.policies.read_permission,
                core_cfg.security.policies.write_permission,
            ),
            ttl_s=600.0,
        )
        input_state["access_token"] = token
        logger.debug(
            "Security enabled (ingestion): minted access_token token_id=%s perms=%s ttl_s=%s",
            getattr(token, "token_id", None),
            list(getattr(token, "permissions", ()) or ()),
            600.0,
        )
    else:
        logger.debug("Security disabled (ingestion): providers will run without token verification")

    # Add langfuse callbacks for ingestion tracing (if available)
    from contextrouter.modules.observability import get_langfuse_callbacks

    callbacks = get_langfuse_callbacks(
        session_id=f"ingestion_{recipe.get('name', 'default') if isinstance(recipe, dict) else 'default'}",
        user_id=user_ctx.get("user_id") if user_ctx else None,
        platform="ingestion",
        tags=(
            ["ingestion", recipe.get("name", "default")]
            if isinstance(recipe, dict)
            else ["ingestion"]
        ),
    )
    out = await graph.ainvoke(input_state, config={"callbacks": callbacks})
    if isinstance(out, dict):
        return out
    return {"result": out}


async def stream_ingestion(
    *,
    spec: IngestionJobSpec | None = None,
    recipe: IngestionRecipe | None = None,
    ingestion_config_path: str = "",
    only_types: list[str] | None = None,
    overwrite: bool = True,
    workers: int = 1,
    skip_preprocess: bool = False,
    skip_taxonomy: bool = False,
    skip_ontology: bool = False,
    skip_graph: bool = False,
    skip_shadow: bool = False,
    skip_export: bool = False,
    skip_deploy: bool = False,
    skip_report: bool = False,
    deploy_wait: bool = False,
    user_ctx: UserCtx | None = None,
    run_id: str | None = None,
) -> AsyncIterator[IngestionProgressEvent]:
    """Stream normalized progress events for ingestion."""

    rid = run_id or str(uuid.uuid4())
    if isinstance(spec, dict):
        recipe_from_spec, input_state = _input_state_from_spec(spec)
        if recipe_from_spec is not None:
            recipe = recipe_from_spec
    else:
        input_state = _mk_input_state(
            ingestion_config_path=ingestion_config_path,
            only_types=only_types,
            overwrite=overwrite,
            workers=workers,
            skip_preprocess=skip_preprocess,
            skip_taxonomy=skip_taxonomy,
            skip_ontology=skip_ontology,
            skip_graph=skip_graph,
            skip_shadow=skip_shadow,
            skip_export=skip_export,
            skip_deploy=skip_deploy,
            skip_report=skip_report,
            deploy_wait=deploy_wait,
        )

    # Restrict progress events to only the stages present in this run.
    if isinstance(recipe, dict):
        stage_names: set[str] = {str(s) for s in rag_ingestion.resolve_recipe_stages(recipe)}
    else:
        stage_names = {
            "load",
            "preprocess",
            "persona",
            "taxonomy",
            "ontology",
            "graph",
            "shadow",
            "export",
            "deploy",
            "report",
        }

    core_cfg = get_core_config()
    if core_cfg.security.enabled:
        builder = TokenBuilder(enabled=True, private_key_path=core_cfg.security.private_key_path)
        token = builder.mint_root(
            user_ctx=user_ctx or {},
            permissions=(
                core_cfg.security.policies.read_permission,
                core_cfg.security.policies.write_permission,
            ),
            ttl_s=600.0,
        )
        input_state["access_token"] = token

    yield {
        "type": "run_started",
        "run_id": rid,
        "ts": _now(),
        "graph": "rag_ingestion",
    }

    graph = (
        rag_ingestion.compile_graph_from_recipe(recipe)
        if isinstance(recipe, dict)
        else rag_ingestion.compile_graph()
    )

    # Add langfuse callbacks for ingestion tracing (if available)
    from contextrouter.modules.observability import get_langfuse_callbacks

    callbacks = get_langfuse_callbacks(
        session_id=f"ingestion_{recipe.get('name', 'default') if isinstance(recipe, dict) else 'default'}",
        user_id=user_ctx.get("user_id") if user_ctx else None,
        platform="ingestion",
        tags=(
            ["ingestion", recipe.get("name", "default")]
            if isinstance(recipe, dict)
            else ["ingestion"]
        ),
    )

    try:
        async for event in graph.astream_events(
            input_state, config={"callbacks": callbacks}, version="v2"
        ):
            if not isinstance(event, dict):
                continue

            ev_type = event.get("event")
            name = event.get("name")
            if not isinstance(ev_type, str) or not isinstance(name, str):
                continue
            if name not in stage_names:
                continue

            if ev_type == "on_chain_start":
                yield {
                    "type": "stage_started",
                    "run_id": rid,
                    "ts": _now(),
                    "graph": "rag_ingestion",
                    "stage": cast(StageName, name),
                }
            elif ev_type == "on_chain_end":
                yield {
                    "type": "stage_finished",
                    "run_id": rid,
                    "ts": _now(),
                    "graph": "rag_ingestion",
                    "stage": cast(StageName, name),
                    "status": _extract_stage_status(event, name) or "ok",
                    "output_keys": _extract_output_keys(event),
                }
            elif ev_type == "on_chain_error":
                err = ""
                data = event.get("data")
                if isinstance(data, dict):
                    e = data.get("error")
                    if isinstance(e, str):
                        err = e
                    else:
                        err = str(e)
                yield {
                    "type": "stage_failed",
                    "run_id": rid,
                    "ts": _now(),
                    "graph": "rag_ingestion",
                    "stage": cast(StageName, name),
                    "error": err or "unknown_error",
                }

        yield {
            "type": "run_finished",
            "run_id": rid,
            "ts": _now(),
            "graph": "rag_ingestion",
        }
    except Exception as e:  # noqa: BLE001
        yield {
            "type": "run_failed",
            "run_id": rid,
            "ts": _now(),
            "graph": "rag_ingestion",
            "error": str(e),
        }
        raise


__all__ = ["invoke_ingestion", "stream_ingestion", "IngestionProgressEvent"]
