"""Config Resolution Engine for the Graph Compiler.

Implements 3-level config hierarchy:
    Per-node config → Graph-wide config → Router defaults

Two layers with different rules:

- **Per-node** ``node_spec["config"]`` — validated as :class:`NodeConfig`
  (``extra="forbid"``); unknown keys are author errors.
- **Graph-wide** manifest ``config`` — free-form blob that intentionally mixes
  graph-only keys (``{node}_prompt``, ``visualizer_*``, tool-binding maps, …)
  with node-default knobs. Only the subset matching :class:`NodeConfig` fields
  is merged into each node; the rest is consumed elsewhere (executors read
  it from :class:`~contextunity.router.cortex.compiler.types.ProjectManifest`).
"""

from __future__ import annotations

import logging

from contextunity.core.types import is_object_dict

from contextunity.router.cortex.compiler.node_config import (
    NODE_CONFIG_FIELD_NAMES,
    NodeConfig,
    merge_node_config_dicts,
)
from contextunity.router.cortex.compiler.types import (
    CompilerNodeSpec,
    ProjectManifest,
)

logger = logging.getLogger(__name__)


def _graph_wide_node_defaults(graph_config: ProjectManifest) -> dict[str, object] | None:
    """Return graph-wide *node defaults* (subset of :class:`NodeConfig` fields).

    Accepts either:

    - A full :class:`ProjectManifest` (``nodes`` / ``edges`` present): the
      relevant blob is ``graph_config["config"]``.
    - A flat dict (tests / callers that already unwrapped graph defaults): the
      dict itself is the blob.

    Graph-only keys in the blob (prompts, visualizer, …) are intentionally
    ignored here — they're consumed by executors via ``ProjectManifest``.
    """
    if graph_config.get("nodes") is not None or graph_config.get("edges") is not None:
        blob = graph_config.get("config")
        if blob is None:
            return None
    elif is_object_dict(graph_config.get("config")):
        blob = graph_config.get("config")
    else:
        blob = graph_config

    if not is_object_dict(blob):
        return None
    return {str(k): v for k, v in blob.items() if str(k) in NODE_CONFIG_FIELD_NAMES} or None


def resolve_node_config(
    node_spec: CompilerNodeSpec,
    graph_config: ProjectManifest,
    router_defaults: NodeConfig,
) -> dict[str, object]:
    """Merge config from 3 levels: per-node → graph-wide → Router defaults.

    Priority (highest first):
        1. ``node_spec["config"]`` — per-node settings (strict :class:`NodeConfig`).
        2. Graph-wide defaults — see :func:`_graph_wide_node_defaults`.
        3. ``router_defaults`` — Router-level defaults from :class:`RouterSection`.

    ``tool_config`` and ``provider_config`` dicts are deep-merged across layers.
    """
    merged: dict[str, object] = dict(router_defaults.as_manifest_dict())

    graph_layer = _graph_wide_node_defaults(graph_config)
    if graph_layer:
        merged = merge_node_config_dicts(
            merged,
            NodeConfig.from_mapping(graph_layer).as_manifest_dict(),
        )

    node_level = node_spec.get("config")
    if is_object_dict(node_level) and node_level:
        merged = merge_node_config_dicts(
            merged,
            NodeConfig.from_mapping(node_level).as_manifest_dict(),
        )

    return NodeConfig.model_validate(merged).as_manifest_dict()


def resolve_model(
    node_spec: CompilerNodeSpec,
    graph_config: ProjectManifest,
    router_defaults: NodeConfig,
) -> str:
    """Resolve model with fallback chain.

    Priority:
        1. ``node_spec["model"]``
        2. ``graph_config["default_model"]`` (top-level manifest field)
        3. ``router_defaults.default_model``
        4. ``""`` when nothing is configured
    """
    model = node_spec.get("model")
    if isinstance(model, str) and model.strip():
        return model

    graph_level = graph_config or {}
    model = graph_level.get("default_model")
    if isinstance(model, str) and model.strip():
        return model

    dm = router_defaults.default_model
    if dm:
        return dm

    return ""


def resolve_model_secret_ref(
    node_spec: CompilerNodeSpec,
    graph_config: ProjectManifest,
    router_defaults: NodeConfig,
) -> str | None:
    """Resolve model secret reference with fallback chain.

    Priority:
        1. ``node_spec["model_secret_ref"]``
        2. ``graph_config["default_model_secret_ref"]``
        3. ``router_defaults.default_model_secret_ref``
        4. ``None``
    """
    ref = node_spec.get("model_secret_ref")
    if isinstance(ref, str) and ref.strip():
        return ref

    graph_level = graph_config or {}
    ref = graph_level.get("default_model_secret_ref")
    if isinstance(ref, str) and ref.strip():
        return ref

    dsr = router_defaults.default_model_secret_ref
    if dsr:
        return dsr

    return None


def get_router_defaults() -> NodeConfig:
    """Extract Router-level defaults from :class:`RouterSection`."""
    raw: dict[str, object] = {}
    try:
        from contextunity.router.core.config.models import RouterSection

        config = RouterSection()
        for attr in ("default_model", "default_model_secret_ref"):
            val = getattr(config, attr, None)
            if val:
                raw[attr] = val
    except (ImportError, AttributeError):
        logger.debug("RouterSection not available — using empty defaults")

    return NodeConfig.model_validate(raw)


__all__ = [
    "resolve_node_config",
    "resolve_model",
    "resolve_model_secret_ref",
    "get_router_defaults",
]
