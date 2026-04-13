"""Helper for resolving node configuration dynamically from manifest."""

from typing import Any

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)


def get_node_config(project_config: dict[str, Any], node_name: str) -> dict[str, Any]:
    """Resolve per-node config from the nodes array.

    Low-level helper — operates on project_config dict directly.
    Returns the node dict or empty dict if not found.
    """
    for node in project_config.get("nodes", []):
        if node.get("name") == node_name:
            return node
    return {}


def get_node_attr(
    project_config: dict[str, Any], node_name: str, attr: str, default: Any = None
) -> Any:
    """Get a specific attribute from a named node."""
    return get_node_config(project_config, node_name).get(attr, default)


def get_node_manifest_config(state: dict, node_name: str) -> dict[str, Any]:
    """Extract manifest configuration for a specific node from execution state.

    Convenience wrapper — unwraps ``state → metadata → project_config`` then
    delegates to :func:`get_node_config`.

    Returns a dictionary containing manifest keys like:
        - model
        - model_secret_ref
        - prompt_ref
        - tools
        - pii_masking
    """
    metadata = state.get("metadata", {})
    project_config = metadata.get("project_config", {})
    result = get_node_config(project_config, node_name)
    if not result:
        logger.warning("Node '%s' not found in manifest configuration. Using defaults.", node_name)
    return result


def make_shield_path(node_name: str) -> str:
    """Compute the contextunity.shield lookup path for a node's primary model key."""
    return f"{node_name}/model_secret_ref"
