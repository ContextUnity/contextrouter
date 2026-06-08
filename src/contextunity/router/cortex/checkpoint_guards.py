"""LangGraph checkpoint shape guards for untrusted persistence payloads."""

from __future__ import annotations

from typing import TypeGuard

from contextunity.core.types import is_object_dict, is_object_list
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, PendingWrite

_CHECKPOINT_SOURCES = frozenset({"input", "loop", "update", "fork"})
_CHANNEL_VERSION_SCALAR = (str, int, float)


def _is_channel_versions(value: object) -> bool:
    """Validate ``ChannelVersions`` map values from Redis JSON."""
    if not is_object_dict(value):
        return False
    return all(isinstance(version, _CHANNEL_VERSION_SCALAR) for version in value.values())


def _is_versions_seen(value: object) -> bool:
    """Validate ``versions_seen`` nested channel-version maps."""
    if not is_object_dict(value):
        return False
    return all(_is_channel_versions(inner) for inner in value.values())


def _is_checkpoint(value: object) -> TypeGuard[Checkpoint]:
    """Validate LangGraph ``Checkpoint`` shape from untrusted Redis payloads."""
    if not is_object_dict(value):
        return False
    if not isinstance(value.get("v"), int):
        return False
    if not isinstance(value.get("id"), str):
        return False
    if not isinstance(value.get("ts"), str):
        return False
    if not is_object_dict(value.get("channel_values")):
        return False
    if not _is_channel_versions(value.get("channel_versions")):
        return False
    if not _is_versions_seen(value.get("versions_seen")):
        return False
    updated_channels = value.get("updated_channels")
    if updated_channels is not None:
        if not is_object_list(updated_channels):
            return False
        for channel in updated_channels:
            if not isinstance(channel, str):
                return False
    return True


def _is_checkpoint_metadata(value: object) -> TypeGuard[CheckpointMetadata]:
    """Validate LangGraph ``CheckpointMetadata``; unknown keys are ignored."""
    if not is_object_dict(value):
        return False
    source = value.get("source")
    if source is not None and (not isinstance(source, str) or source not in _CHECKPOINT_SOURCES):
        return False
    step = value.get("step")
    if step is not None and not isinstance(step, int):
        return False
    run_id = value.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        return False
    parents = value.get("parents")
    if parents is not None:
        if not is_object_dict(parents):
            return False
        if not all(isinstance(val, str) for _key, val in parents.items()):
            return False
    return True


def _is_pending_write(value: object) -> TypeGuard[PendingWrite]:
    """Validate LangGraph ``PendingWrite``: ``(task_id, channel, value)``."""
    match value:
        case (str(), str(), object()):
            return True
        case _:
            return False


def _is_pending_writes(value: object) -> TypeGuard[list[PendingWrite]]:
    """Validate a list of pending write tuples."""
    return is_object_list(value) and all(_is_pending_write(item) for item in value)


__all__ = [
    "_is_checkpoint",
    "_is_checkpoint_metadata",
    "_is_pending_write",
    "_is_pending_writes",
]
