"""Visualizer node — formats SQL results into UI components.

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

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from contextrouter.cortex.graphs.sql_analytics.helpers import (
    StepTimer,
    acc_tokens,
    extract_json,
    invoke_model,
    step,
)
from contextrouter.cortex.graphs.sql_analytics.pii import pii_anonymize, pii_deanonymize
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.models import model_registry

logger = logging.getLogger(__name__)


# ── Heuristics ──────────────────────────────────────────────────────


def _needs_chart(rows: list, columns: list) -> bool:
    """Charts are useful when there are numeric columns and >1 row."""
    if len(rows) < 2:
        return False
    numeric_count = 0
    for col in columns:
        for r in rows[:5]:
            v = r.get(col) if isinstance(r, dict) else None
            if v is not None:
                try:
                    float(v)
                    numeric_count += 1
                    break
                except (ValueError, TypeError):
                    pass
    return numeric_count >= 1


def _needs_table(rows: list) -> bool:
    """Tables are useful if there's data."""
    return len(rows) > 0


# ── Node factory ────────────────────────────────────────────────────


def make_visualizer_node(
    *,
    visualizer_prompt: str | None,
    visualizer_sub_prompts: dict[str, str] | None = None,
    default_model_key: str | None,
    pii_masking: bool = False,
    anonymize_tool: BaseTool | None = None,
    deanonymize_tool: BaseTool | None = None,
):
    """Create the visualizer node closure.

    Args:
        visualizer_prompt: Single system prompt (fallback / single mode).
        visualizer_sub_prompts: Optional dict ``{"report": ..., "table": ...,
            "chart": ...}`` for parallel execution.
        default_model_key: Fallback model key.
        pii_masking: Whether to anonymize data before sending to LLM.
        anonymize_tool: Tool for PII anonymization.
        deanonymize_tool: Tool for PII deanonymization.
    """
    # Decide mode at build time
    use_parallel = bool(
        visualizer_sub_prompts
        and isinstance(visualizer_sub_prompts, dict)
        and "report" in visualizer_sub_prompts
    )

    if use_parallel:
        logger.info(
            "Visualizer: parallel mode (sub-prompts: %s)", list(visualizer_sub_prompts.keys())
        )
    else:
        logger.info("Visualizer: single-call mode")

    async def visualizer_node(state: SqlAnalyticsState):
        sql_result = state.get("sql_result") or {}
        rows = sql_result.get("rows", []) if isinstance(sql_result, dict) else []
        columns = sql_result.get("columns", []) if isinstance(sql_result, dict) else []

        if not visualizer_prompt and not use_parallel:
            col_objs = [{"key": c, "label": c} for c in columns] if columns else []
            return {
                "components": [{"type": "table", "columns": col_objs, "rows": rows[:200]}],
                "_steps": [step("visualizer", status="skipped", reason="no_prompt")],
            }

        metadata = state.get("metadata") or {}

        # User question context
        user_q = "Results available."
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                user_q = m.content
                break

        # Build shared data context
        summary = _compute_summary(rows, columns)
        if len(rows) <= 50:
            data_rows = rows
            data_note = f"All {len(rows)} rows"
        else:
            data_rows = rows[:20]
            data_note = f"Sample: {len(data_rows)} of {len(rows)} rows"

        data_context = _build_data_context(user_q, columns, rows, data_rows, data_note, summary)

        # ── PII anonymize shared context ──
        session_id = metadata.get("session_id", "")
        sub_steps: list[dict] = []
        if pii_masking:
            anon_timer = StepTimer()
            ctx_before = data_context[:200]
            with anon_timer:
                data_context = await pii_anonymize(
                    data_context,
                    tool=anonymize_tool,
                    session_id=session_id,
                )
            sub_steps.append(
                step(
                    "pii_anonymize",
                    timer=anon_timer,
                    request={"context_preview": ctx_before},
                    result={"masked_preview": data_context[:200]},
                )
            )

        model_key = metadata.get("model_key") or default_model_key
        llm = model_registry.get_llm_with_fallback(key=model_key)

        timer = StepTimer()
        try:
            if use_parallel:
                comps, usage, sub_steps = await _parallel_path(
                    llm,
                    visualizer_sub_prompts,
                    data_context,
                    rows,
                    columns,
                    timer,
                    sub_steps,
                    model_key,
                )
            else:
                comps, usage, sub_steps = await _single_path(
                    llm,
                    visualizer_prompt,
                    data_context,
                    rows,
                    user_q,
                    timer,
                    sub_steps,
                    model_key,
                )

            # ── PII deanonymize after generation ──
            if pii_masking and comps:
                deanon_timer = StepTimer()
                count_before = len(comps)
                with deanon_timer:
                    comps = await _deanonymize_components(
                        comps,
                        deanonymize_tool,
                        session_id,
                    )
                sub_steps.append(
                    step(
                        "pii_deanonymize",
                        timer=deanon_timer,
                        request={"components_count": count_before},
                        result={"components_count": len(comps)},
                    )
                )

            return {
                "components": comps,
                "messages": [
                    AIMessage(
                        content=json.dumps({"components": comps}, ensure_ascii=False, default=str)[
                            :2000
                        ]
                    )
                ],
                "_token_usage": acc_tokens(state, usage),
                "_steps": sub_steps,
            }
        except Exception as e:
            logger.error("Visualizer failed: %s", e)
            sub_steps.append(
                step("visualizer", status="error", timer=timer if "timer" in dir() else None)
            )
            col_objs = [{"key": c, "label": c} for c in columns] if columns else []
            return {
                "components": [{"type": "table", "columns": col_objs, "rows": rows[:200]}],
                "_steps": sub_steps,
            }

    return visualizer_node


# ── Execution paths ─────────────────────────────────────────────────


async def _single_path(
    llm: Any,
    prompt: str,
    data_context: str,
    rows: list,
    user_q: str,
    timer: StepTimer,
    sub_steps: list[dict],
    model_key: str | None,
) -> tuple[list[dict], dict, list[dict]]:
    """Single LLM call path."""
    messages = [SystemMessage(content=prompt), HumanMessage(content=data_context)]

    with timer:
        response, usage = await invoke_model(llm, messages)

    data = extract_json(response.content)
    comps = data.get("components", []) if data else []

    # Inject full rows into table components (LLM only saw a sample)
    for comp in comps:
        if isinstance(comp, dict) and comp.get("type") == "table" and rows:
            comp["rows"] = rows[:200]

    comp_types = [c.get("type") for c in comps if isinstance(c, dict)][:5]
    sub_steps.append(
        step(
            "visualizer",
            timer=timer,
            request={"question": user_q[:500], "row_count": len(rows)},
            result={"components": comp_types, "count": len(comps)},
            components=len(comps),
        )
    )
    return comps, usage, sub_steps


async def _parallel_path(
    llm: Any,
    sub_prompts: dict[str, str],
    data_context: str,
    rows: list,
    columns: list,
    timer: StepTimer,
    sub_steps: list[dict],
    model_key: str | None,
) -> tuple[list[dict], dict, list[dict]]:
    """Parallel LLM calls path — runs report/table/chart concurrently."""
    # Determine which sub-calls to run
    tasks: dict[str, str] = {"report": sub_prompts["report"]}
    if _needs_table(rows) and "table" in sub_prompts:
        tasks["table"] = sub_prompts["table"]
    if _needs_chart(rows, columns) and "chart" in sub_prompts:
        tasks["chart"] = sub_prompts["chart"]

    with timer:
        results = await _run_parallel_calls(llm, tasks, data_context)

    # Record sub-steps for each parallel LLM call
    for call_name, (comps, usage) in results.items():
        sub_steps.append(
            step(
                f"llm_{call_name}",
                timer=timer,  # shares wall-clock (parallel)
                request={"prompt": call_name, "model": str(model_key)},
                result={
                    "components": len(comps),
                    "types": [c.get("type") for c in comps if isinstance(c, dict)][:5],
                },
                tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                cost_usd=usage.get("total_cost", 0),
            )
        )

    # Merge components in order and accumulate usage
    all_components, total_usage = _merge_results(results, rows)

    merge_step = step(
        "component_merge",
        timer=timer,
        request={"parallel_calls": list(tasks.keys())},
        result={
            "total_components": len(all_components),
            "sub_calls": {k: len(v[0]) for k, v in results.items()},
        },
    )
    sub_steps.append(merge_step)

    return all_components, total_usage, sub_steps


# ── Internal helpers ────────────────────────────────────────────────


def _compute_summary(rows: list, columns: list) -> dict:
    """Compute summary stats for numeric columns."""
    summary: dict[str, Any] = {}
    if not rows or not columns:
        return summary
    for col in columns:
        nums = []
        for r in rows:
            v = r.get(col) if isinstance(r, dict) else None
            if v is not None:
                try:
                    nums.append(float(v))
                except (ValueError, TypeError):
                    pass
        if nums:
            summary[col] = {
                "count": len(nums),
                "sum": round(sum(nums), 2),
                "min": round(min(nums), 2),
                "max": round(max(nums), 2),
                "avg": round(sum(nums) / len(nums), 2),
            }
    return summary


def _build_data_context(
    user_q: str,
    columns: list,
    rows: list,
    data_rows: list,
    data_note: str,
    summary: dict,
) -> str:
    """Build the shared data context string for LLM prompts."""
    parts = [
        f"User Question: {user_q}",
        f"Columns: {json.dumps(columns, default=str)}",
        f"Total rows: {len(rows)}",
    ]
    if summary:
        parts.append(f"Column stats (for KPI cards): {json.dumps(summary, ensure_ascii=False)}")
    parts.append(
        f"Data ({data_note}): {json.dumps(data_rows, ensure_ascii=False, default=str)[:4000]}"
    )
    return "\n".join(parts)


async def _run_parallel_calls(
    llm: Any,
    tasks: dict[str, str],
    data_context: str,
) -> dict[str, tuple[list[dict], dict]]:
    """Run sub-calls in parallel and return {name: (components, usage)}."""

    async def _one_call(name: str, system_prompt: str) -> tuple[str, list[dict], dict]:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=data_context)]
        response, usage = await invoke_model(llm, messages)
        data = extract_json(response.content)
        comps = data.get("components", []) if data else []
        return name, comps, usage

    coros = [_one_call(name, prompt) for name, prompt in tasks.items()]
    results_list = await asyncio.gather(*coros, return_exceptions=True)

    results: dict[str, tuple[list[dict], dict]] = {}
    for item in results_list:
        if isinstance(item, Exception):
            logger.warning("Visualizer sub-call failed: %s", item)
            continue
        name, comps, usage = item
        results[name] = (comps, usage)

    return results


def _merge_results(
    results: dict[str, tuple[list[dict], dict]],
    rows: list,
) -> tuple[list[dict], dict]:
    """Merge parallel results into ordered component list + total usage."""
    all_components: list[dict] = []
    total_usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}

    # Collect from each sub-call in order: report → chart → table
    for call_name in ("report", "chart", "table"):
        if call_name not in results:
            continue
        comps, usage = results[call_name]
        _acc_usage(total_usage, usage)

        for comp in comps:
            if not isinstance(comp, dict):
                continue
            # Inject full rows into table (LLM only generated sample rows)
            if call_name == "table" and comp.get("type") == "table" and rows:
                comp["rows"] = rows[:200]
            all_components.append(comp)

    # Reorder: title → KPI → table → chart → conclusion → other
    header = [
        c
        for c in all_components
        if c.get("type") == "text" and "Висновок" not in str(c.get("content", ""))
    ]
    kpis = [c for c in all_components if c.get("type") == "kpi_cards"]
    charts = [
        c for c in all_components if c.get("type") in ("bar_chart", "line_chart", "pie_chart")
    ]
    tables = [c for c in all_components if c.get("type") == "table"]
    conclusions = [
        c
        for c in all_components
        if c.get("type") == "text" and "Висновок" in str(c.get("content", ""))
    ]
    categorised = set(id(c) for lst in (header, kpis, charts, tables, conclusions) for c in lst)
    other = [c for c in all_components if id(c) not in categorised]

    ordered = header + kpis + tables + charts + conclusions + other
    return ordered, total_usage


async def _deanonymize_components(
    components: list[dict],
    deanonymize_tool: BaseTool | None,
    session_id: str,
) -> list[dict]:
    """Single-pass PII deanonymize on merged components."""
    if not deanonymize_tool or not session_id:
        return components
    try:
        raw = json.dumps(components, ensure_ascii=False, default=str)
        restored = await pii_deanonymize(raw, tool=deanonymize_tool, session_id=session_id)
        parsed = json.loads(restored)
        if isinstance(parsed, list):
            return parsed
        logger.warning("deanonymize_components: expected list, got %s", type(parsed))
    except Exception as e:
        logger.warning("deanonymize_components failed, using originals: %s", e)
    return components


def _acc_usage(total: dict, usage: dict) -> None:
    """Accumulate usage dict in-place."""
    total["input_tokens"] += usage.get("input_tokens", 0)
    total["output_tokens"] += usage.get("output_tokens", 0)
    total["total_cost"] += usage.get("total_cost", 0.0)


__all__ = ["make_visualizer_node"]
