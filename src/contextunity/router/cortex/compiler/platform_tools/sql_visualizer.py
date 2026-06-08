"""SQL Visualizer node — formats SQL results into UI components.
Supports two execution modes:
1. **Parallel mode** (fast, ~10-15s) — when project provides ``visualizer_sub_prompts``
   dict with focused prompts for report / table / chart.  Each sub-call runs
   concurrently via ``asyncio.gather``, results are merged.
2. **Single mode** (slower, ~30-50s) — falls back to one LLM call with
   ``visualizer_prompt`` when sub-prompts are not provided.
The prompts themselves are business logic — they come from the project
registration config, never hardcoded here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import ClassVar, Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.compiler.platform_tools.helpers.parallel import (
    ParallelCallResult,
    ParsedComponent,
    run_parallel_llm_calls,
)
from contextunity.router.cortex.compiler.platform_tools.helpers.sql import (
    JsonMap,
    TokenUsageDict,
    acc_tokens,
    build_data_context,
    compute_column_summary,
    extract_json,
    invoke_model,
    needs_chart,
    needs_table,
)
from contextunity.router.cortex.compiler.state_routing import read_state_input
from contextunity.router.cortex.config_resolution import get_node_config
from contextunity.router.cortex.types import (
    GraphState,
    NodeFunc,
    RegisteredProjectConfig,
    StateUpdate,
    extract_message_content,
    is_registered_project_config,
)
from contextunity.router.modules.models import model_registry
from contextunity.router.modules.models.base import BaseLLM as LLMBaseModel

# Reusable row/column types for narrowing
RowList = list[JsonMap]
ColList = list[str]
ModelInvokeFunc = Callable[..., Awaitable[tuple[AIMessage, TokenUsageDict]]]


def _narrow_dict(obj: object) -> JsonMap:
    """Coerce *obj* to a ``dict[str, object]``; return an empty dict if *obj* is not a mapping."""
    if is_object_dict(obj):
        return dict(obj)
    return {}


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    """Return a mapping value as ``str`` when present."""
    value = mapping.get(key)
    return str(value) if value is not None else None


def _coerce_components(value: object) -> list[ParsedComponent]:
    """Coerce raw JSON payloads into parsed visualizer components."""
    if not is_object_list(value):
        return []

    components: list[ParsedComponent] = []
    for item in value:
        if not is_object_dict(item):
            continue
        component_type = item.get("type")
        if not isinstance(component_type, str):
            continue
        component: ParsedComponent = {"type": component_type}
        for key in ("title", "content"):
            text_value = item.get(key)
            if isinstance(text_value, str):
                component[key] = text_value
        if "rows" in item and is_object_list(item["rows"]):
            component["rows"] = [dict(row) for row in item["rows"] if is_object_dict(row)]
        if "columns" in item and is_object_list(item["columns"]):
            component["columns"] = [
                {column_key: str(column_value) for column_key, column_value in column.items()}
                for column in item["columns"]
                if is_object_dict(column)
            ]
        components.append(component)
    return components


class SqlVisualizerConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    default_format: Literal["table", "chart", "markdown"] = "table"
    max_chart_points: int = Field(default=100, ge=1, le=10000)


logger = get_contextunit_logger(__name__)

# ── Node factory ─────────────────────────────────────────────────────


def make_visualizer_node(
    *,
    node_name: str = "visualizer",
    visualizer_prompt: str | None,
    visualizer_sub_prompts: dict[str, str] | None = None,
    default_model_key: str | None,
    fallback_keys: list[str] | None = None,
    shield_key_name: str | None = None,
    invoke_model_fn: ModelInvokeFunc = invoke_model,
) -> NodeFunc:
    """Create the sql visualizer node closure."""
    _ = shield_key_name
    use_parallel = bool(visualizer_sub_prompts and "report" in visualizer_sub_prompts)

    async def visualizer_node(state: GraphState, config: RunnableConfig) -> StateUpdate:
        """Format SQL query results into UI components (report/table/chart) via one or more LLM calls."""
        # ── Extract SQL result from state ────────────────────────────
        sql_result = _extract_sql_result(state)
        raw_rows = sql_result.get("rows", [])
        raw_cols = sql_result.get("columns", [])
        rows: RowList = (
            [dict(row) for row in raw_rows if is_object_dict(row)]
            if is_object_list(raw_rows)
            else []
        )
        columns: ColList = [str(column) for column in raw_cols] if is_object_list(raw_cols) else []

        logger.info(
            "visualizer_node: sql_result keys=%s, rows=%d, columns=%d, needs_chart=%s, needs_table=%s",
            list(sql_result.keys()),
            len(rows),
            len(columns),
            needs_chart(rows, columns),
            needs_table(rows),
        )

        # Fast path: no prompts → raw table fallback
        if not visualizer_prompt and not use_parallel:
            return _make_table_fallback(columns, rows)

        # ── Build data context for LLM ───────────────────────────────
        metadata = state.get("metadata") or {}
        user_q = _extract_user_question(state)
        summary = compute_column_summary(rows, columns)

        if len(rows) <= 50:
            data_rows, data_note = rows, f"All {len(rows)} rows"
        else:
            data_rows = rows[:20]
            data_note = f"Sample: {len(data_rows)} of {len(rows)} rows"

        data_context = build_data_context(user_q, columns, rows, data_rows, data_note, summary)

        # PII is applied by the injected LLM-node invoker, after prompt assembly.

        # Token is always injected into state by secure_node — guaranteed non-None.
        from contextunity.router.cortex.compiler.platform_tools.helpers.base import (
            resolve_tenant_from_state,
        )

        tenant_id = resolve_tenant_from_state(state, binding="router_sql_visualizer")

        create_kwargs: dict[str, str] = {}
        if tenant_id:
            create_kwargs["tenant_id"] = tenant_id

        llm = model_registry.get_llm_with_fallback(
            default_model_key,
            fallback_keys=fallback_keys,
        )

        try:
            project_config: RegisteredProjectConfig = {}
            project_config_raw = metadata.get("project_config")
            if is_registered_project_config(project_config_raw):
                project_config = project_config_raw
            planner_intent = _extract_planner_intent(state)

            if use_parallel:
                if visualizer_sub_prompts is None:
                    from contextunity.core.exceptions import ConfigurationError

                    raise ConfigurationError("visualizer_sub_prompts required in parallel mode.")
                comps, usage = await _parallel_path(
                    llm,
                    visualizer_sub_prompts,
                    data_context,
                    rows,
                    columns,
                    config=config,
                    project_config=project_config,
                    planner_intent=planner_intent,
                    invoke_model_fn=invoke_model_fn,
                    state=state,
                    node_name=node_name,
                )
            else:
                if visualizer_prompt is None:
                    from contextunity.core.exceptions import ConfigurationError

                    raise ConfigurationError("visualizer_prompt required in single mode.")
                comps, usage = await _single_path(
                    llm,
                    visualizer_prompt,
                    data_context,
                    rows,
                    config=config,
                    state=state,
                    invoke_model_fn=invoke_model_fn,
                    node_name=node_name,
                )

            result_payload = {"components": comps}
            return {
                "components": comps,
                "final_output": result_payload,
                "messages": [
                    AIMessage(
                        content=json_dumps(result_payload, ensure_ascii=False, default=str)[:2000]
                    )
                ],
                "_token_usage": acc_tokens(state, usage),
            }
        except Exception as e:  # wraps-to-domain: re-raises as typed exception
            from contextunity.core.exceptions import ContextUnityError

            if isinstance(e, ContextUnityError):
                raise
            logger.error("Visualizer failed: %s", e)
            return _make_table_fallback(columns, rows)

    return visualizer_node


# ── Execution paths ──────────────────────────────────────────────────


async def _single_path(
    llm: LLMBaseModel,
    prompt: str,
    data_context: str,
    rows: RowList,
    config: RunnableConfig,
    state: GraphState | None = None,
    invoke_model_fn: ModelInvokeFunc = invoke_model,
    node_name: str = "visualizer",
) -> tuple[list[ParsedComponent], TokenUsageDict]:
    """Run a single LLM call with the full visualizer prompt and return parsed components."""
    messages = [SystemMessage(content=prompt), HumanMessage(content=data_context)]
    response, usage = await invoke_model_fn(
        llm, messages, config=config, node_name=node_name, state=state
    )
    raw_content = extract_message_content(response)
    data = extract_json(raw_content)
    comps = _coerce_components(data.get("components", []) if data else [])
    for comp in comps:
        if comp.get("type") == "table" and rows:
            comp["rows"] = rows[:200]
    return comps, usage


async def _parallel_path(
    llm: LLMBaseModel,
    sub_prompts: dict[str, str],
    data_context: str,
    rows: RowList,
    columns: ColList,
    config: RunnableConfig,
    project_config: RegisteredProjectConfig,
    planner_intent: str = "",
    invoke_model_fn: ModelInvokeFunc = invoke_model,
    state: GraphState | None = None,
    node_name: str = "visualizer",
) -> tuple[list[ParsedComponent], TokenUsageDict]:
    """Fan out report/table/chart sub-prompts as concurrent LLM calls and merge results."""
    visualizer_node = get_node_config(project_config, "visualizer")
    variants_versions = _narrow_dict(visualizer_node.get("prompt_variants_versions", {}))

    tasks: dict[str, tuple[str, str | None]] = {
        "report": (
            sub_prompts["report"],
            _optional_string(variants_versions, "report"),
        )
    }

    intent = planner_intent.strip().lower()

    # Always dispatch table/chart when data supports it.
    # Only "answer" intent explicitly skips both (simple text response).
    skip_visuals = intent == "answer"

    if not skip_visuals and needs_table(rows) and "table" in sub_prompts:
        tasks["table"] = (
            sub_prompts["table"],
            _optional_string(variants_versions, "table"),
        )
    if not skip_visuals and needs_chart(rows, columns) and "chart" in sub_prompts:
        tasks["chart"] = (
            sub_prompts["chart"],
            _optional_string(variants_versions, "chart"),
        )

    logger.info(
        "Visualizer parallel dispatch: intent=%s → tasks=%s",
        intent or "(heuristic)",
        list(tasks.keys()),
    )

    results = await run_parallel_llm_calls(
        llm,
        tasks,
        data_context,
        config,
        invoke_model_fn=invoke_model_fn,
        state=state,
        node_name=node_name,
    )

    logger.info(
        "Visualizer parallel results: dispatched=%s, returned=%s",
        list(tasks.keys()),
        {k: [c.get("type") for c in comps] for k, (comps, _) in results.items()},
    )

    return _merge_results(results, rows)


# ── Result merging ───────────────────────────────────────────────────


def _merge_results(
    results: dict[str, ParallelCallResult],
    rows: RowList,
) -> tuple[list[ParsedComponent], TokenUsageDict]:
    """Merge parallel call results into an ordered component list (text → KPIs → tables → charts) plus total token usage."""
    all_components: list[ParsedComponent] = []
    total_usage: TokenUsageDict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }

    for call_name in ("report", "chart", "table"):
        if call_name not in results:
            continue
        comps, usage = results[call_name]
        total_usage["input_tokens"] += int(usage.get("input_tokens", 0))
        total_usage["output_tokens"] += int(usage.get("output_tokens", 0))
        total_usage["total_tokens"] += int(usage.get("total_tokens", 0))
        total_usage["total_cost"] += float(usage.get("total_cost", 0.0))

        for comp in comps:
            if call_name == "table" and comp.get("type") == "table" and rows:
                comp["rows"] = rows[:200]
            all_components.append(comp)

    # Order: text → KPIs → tables → charts → conclusions → other
    _CHART_TYPES = {"bar_chart", "line_chart", "pie_chart"}
    header = [c for c in all_components if c.get("type") == "text"]
    kpis = [c for c in all_components if c.get("type") == "kpi_cards"]
    charts = [c for c in all_components if c.get("type") in _CHART_TYPES]
    tables = [c for c in all_components if c.get("type") == "table"]
    conclusions = [c for c in all_components if c.get("type") == "conclusion"]
    categorised = {id(c) for lst in (header, kpis, charts, tables, conclusions) for c in lst}
    other = [c for c in all_components if id(c) not in categorised]

    return header + kpis + tables + charts + conclusions + other, total_usage


# ── State extraction helpers ─────────────────────────────────────────


def _extract_sql_result(state: GraphState) -> JsonMap:
    """Extract sql result dict from state via routing layer."""
    result = _narrow_dict(read_state_input(state, "sql_result", default={}))
    if not result:
        intermediate = _narrow_dict(state.get("intermediate_results") or {})
        for key in ("tool_execution", "execute_sql"):
            cand = _narrow_dict(intermediate.get(key))
            if cand.get("rows"):
                return cand
    if not result:
        fo = _narrow_dict(state.get("final_output") or {})
        if fo.get("rows"):
            return fo
    return result


def _extract_user_question(state: GraphState) -> str:
    """Extract original user question from message history."""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return extract_message_content(m)
    return "Results available."


def _extract_planner_intent(state: GraphState) -> str:
    """Extract planner format intent from state via routing layer."""
    planner_intent = str(read_state_input(state, "format", default="") or "")
    if not planner_intent:
        intermediate = _narrow_dict(state.get("intermediate_results") or {})
        planner_out = _narrow_dict(intermediate.get("planner"))
        planner_intent = _optional_string(planner_out, "format") or ""
    if not planner_intent:
        fo = _narrow_dict(state.get("final_output") or {})
        planner_intent = _optional_string(fo, "format") or ""
    return planner_intent


def _make_table_fallback(columns: ColList, rows: RowList) -> StateUpdate:
    """Create a raw table fallback when no llm prompts are available."""
    col_objs = [{"key": c, "label": c} for c in columns] if columns else []
    fallback: list[ParsedComponent] = [{"type": "table", "columns": col_objs, "rows": rows[:200]}]
    return {
        "components": fallback,
        "final_output": {"components": fallback},
    }


__all__ = ["make_visualizer_node"]
