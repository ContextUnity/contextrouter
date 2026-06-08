"""Helpers for building typed dispatcher execution metadata from wire payloads."""

from __future__ import annotations

from contextunity.core.types import JsonDict

from contextunity.router.cortex.types import ExecutionMetadata, RegisteredProjectConfig


def execution_metadata_from_payload(
    meta: JsonDict,
    *,
    model_key: str | None = None,
    project_id: str | None = None,
    project_config: RegisteredProjectConfig | None = None,
    system_prompt: str | None = None,
) -> ExecutionMetadata:
    """Narrow L2 ``JsonDict`` client metadata into ``ExecutionMetadata``."""
    result: ExecutionMetadata = {}
    langfuse = meta.get("langfuse_enabled")
    if isinstance(langfuse, (bool, str)):
        result["langfuse_enabled"] = langfuse
    for field in (
        "agent_id",
        "graph_name",
        "model_key",
        "system_prompt",
    ):
        val = meta.get(field)
        if isinstance(val, str):
            result[field] = val
    for list_field in ("allowed_tools", "denied_tools"):
        val = meta.get(list_field)
        if isinstance(val, list):
            result[list_field] = [str(t) for t in val if isinstance(t, str)]
    if model_key:
        result["model_key"] = model_key
    if project_id:
        result["project_id"] = project_id
    if system_prompt:
        result["system_prompt"] = system_prompt
    if project_config is not None:
        result["project_config"] = project_config
    return result


def merge_json_metadata(*parts: JsonDict) -> JsonDict:
    """Shallow-merge JSON metadata dicts with L2-valid values only."""
    merged: JsonDict = {}
    for part in parts:
        for key, value in part.items():
            merged[key] = value
    return merged


def copy_execution_metadata(metadata: ExecutionMetadata) -> ExecutionMetadata:
    """Return a shallow copy of execution metadata for safe mutation."""
    copied: ExecutionMetadata = {}
    for key, value in metadata.items():
        copied[key] = value
    return copied


def execution_metadata_for_trace(metadata: ExecutionMetadata) -> dict[str, object]:
    """Widen typed execution metadata for Langfuse trace context."""
    trace: dict[str, object] = {}
    for key, value in metadata.items():
        trace[key] = value
    return trace


__all__ = [
    "copy_execution_metadata",
    "execution_metadata_for_trace",
    "execution_metadata_from_payload",
    "merge_json_metadata",
]
