"""Single parse boundary for per-node manifest ``config`` blocks.
YAML / JSON manifests deserialize to plain dicts. :class:`NodeConfig` lists
every router-known field explicitly. Vendor-specific generation knobs must live
under ``provider_config``; tool executors read ``tool_config`` only.
Unknown top-level keys are rejected (``extra="forbid"``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Self

from contextunity.core.types import is_object_dict
from pydantic import BaseModel, ConfigDict, Field


class NodeConfig(BaseModel):
    """Per-node ``nodes[].config`` and graph-wide node-default layers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    # ── Routing ──
    state_input_key: str | None = None
    state_output_key: str | None = None
    state_input_fields: list[str] | None = None
    output_mode: str | None = None
    output_format: str | None = None
    next_node: str | None = None

    # ── LLM ──
    model: str | None = None
    prompt_ref: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str | None = None
    max_tool_calls: int | None = None
    goal: str | None = None
    persona: str | None = None
    system_prompt: str | None = None

    # ── Memory / Brain ──
    experience_lookup: bool | None = None
    experience_min_q: float | None = None
    experience_limit: int | None = None

    # ── Federated ──
    timeout: int | None = None

    # ── Graph-level defaults (merged into every node) ──
    default_model: str | None = None
    default_model_secret_ref: str | None = None
    data_sources: list[dict[str, str]] | None = None

    tool_config: dict[str, object] = Field(default_factory=dict)
    provider_config: dict[str, object] = Field(default_factory=dict)
    retry: dict[str, object] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> Self:
        """Validate *data* through Pydantic; return a default instance when *data* is falsy."""
        if not data:
            return cls()
        return cls.model_validate(dict(data))

    def as_manifest_dict(self) -> dict[str, object]:
        """Export non-None fields as a plain dict suitable for manifest merge layers."""
        data = self.model_dump(mode="python", exclude_none=True)
        if data.get("tool_config") == {}:
            data.pop("tool_config", None)
        if data.get("provider_config") == {}:
            data.pop("provider_config", None)
        return data


NODE_CONFIG_FIELD_NAMES: frozenset[str] = frozenset(NodeConfig.model_fields.keys())


def merge_node_config_dicts(
    base: Mapping[str, object], overlay: Mapping[str, object]
) -> dict[str, object]:
    """Shallow-merge *overlay* onto *base*; deep-merge ``tool_config`` and ``provider_config``."""

    def _string_object_dict(value: object) -> dict[str, object]:
        if not is_object_dict(value):
            return {}
        return {str(key): item for key, item in value.items()}

    out: dict[str, object] = dict(base)
    for key, val in overlay.items():
        if key == "tool_config" and is_object_dict(val):
            prev = out.get("tool_config")
            val_m = _string_object_dict(val)
            if is_object_dict(prev):
                prev_m = _string_object_dict(prev)
                out["tool_config"] = {**prev_m, **val_m}
            else:
                out["tool_config"] = val_m
        elif key == "provider_config" and is_object_dict(val):
            prev = out.get("provider_config")
            val_m = _string_object_dict(val)
            if is_object_dict(prev):
                prev_m = _string_object_dict(prev)
                out["provider_config"] = {**prev_m, **val_m}
            else:
                out["provider_config"] = val_m
        else:
            out[key] = val
    return out


__all__ = ["NODE_CONFIG_FIELD_NAMES", "NodeConfig", "merge_node_config_dicts"]
