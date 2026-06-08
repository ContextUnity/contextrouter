"""Compiled graph state and conditional routing helpers."""

from __future__ import annotations

from collections.abc import Callable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list
from langgraph.types import Send

from contextunity.router.cortex.types import GraphState

logger = get_contextunit_logger(__name__)

# -- Condition return type -----------------------------------------------------

ConditionResult = str | list[Send]
"""Resolved routing target: a single node name or a fan-out list of Sends."""


# -- Condition routing ---------------------------------------------------------


def create_condition(
    condition_key: str,
    cond_map: dict[str, str] | None = None,
) -> Callable[[GraphState], ConditionResult]:
    """Build a LangGraph conditional edge resolver that reads ``final_output[condition_key]``.

    Supports two routing modes:

    1. **Fan-out** — when the resolved value is ``"selected_sources_fanout"``,
       emit a ``Send()`` per selected data source, routing each to the node
       matching its source type (vector/sql/web) in *cond_map*.
    2. **Scalar** — match the resolved value against *cond_map* keys.  Booleans
       are lowercased (``"true"``/``"false"``).

    Fallback cascade when the resolved value is missing or not in *cond_map*:
    ``"default"`` → ``"success"`` → last key in *cond_map*.
    """

    def _condition_func(state: GraphState) -> ConditionResult:
        final = state.get("final_output")
        val = final.get(condition_key) if is_object_dict(final) else None

        if val == "selected_sources_fanout":
            # selected_sources and config come from previous nodes via
            # intermediate_results — runtime-dynamic, need narrowing.
            ir = state.get("intermediate_results", {})
            raw_selected = state.get("selected_sources") or ir.get("selected_sources")
            selected: list[str] = (
                [str(source) for source in raw_selected] if is_object_list(raw_selected) else []
            )

            raw_config = state.get("config")
            config_candidate = raw_config if is_object_dict(raw_config) else ir.get("config")
            config_dict: dict[str, object] = (
                dict(config_candidate) if is_object_dict(config_candidate) else {}
            )
            raw_ds = config_dict.get("data_sources")
            data_sources: list[dict[str, object]] = (
                [dict(ds) for ds in raw_ds if is_object_dict(ds)] if is_object_list(raw_ds) else []
            )

            from contextunity.router.cortex.compiler.template_loader import (
                DataSourceDefinition,
            )

            ds_type_map: dict[str, str] = {}
            for ds in data_sources:
                try:
                    validated_ds = DataSourceDefinition.model_validate(ds)
                    ds_type_map[validated_ds.binding] = validated_ds.type
                except Exception as exc:  # graceful-degrade: state merge failure logged
                    logger.warning(
                        "Invalid data source skipped in fan-out: %s | error: %s",
                        ds,
                        exc,
                    )

            sends: list[Send] = []
            for src in selected:
                ds_type = ds_type_map.get(src, "vector")
                target_node = cond_map.get(ds_type) if cond_map else None
                if target_node:
                    sends.append(Send(target_node, {"__source_binding__": src}))

            return sends if sends else "no_results"

        if isinstance(val, bool):
            resolved: str | None = str(val).lower()
        elif val is not None:
            resolved = str(val)
        else:
            resolved = None

        if cond_map and resolved and resolved in cond_map:
            return resolved

        if cond_map:
            fallback = (
                "default"
                if "default" in cond_map
                else "success"
                if "success" in cond_map
                else list(cond_map.keys())[-1]
            )
            logger.warning(
                "Condition '%s' resolved to '%s' which is not in condition_map %s. Falling back to '%s'.",
                condition_key,
                resolved,
                list(cond_map.keys()),
                fallback,
            )
            return fallback

        return resolved or "default"

    return _condition_func


__all__ = ["ConditionResult", "create_condition"]
