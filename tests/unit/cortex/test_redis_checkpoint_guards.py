"""Unit tests for Redis checkpoint guard helpers."""

from __future__ import annotations

from contextunity.router.cortex.checkpoint_guards import (
    _is_checkpoint,
    _is_checkpoint_metadata,
    _is_pending_write,
)
from contextunity.router.modules.observability.contracts import is_span_dict


def _valid_checkpoint() -> dict[str, object]:
    return {
        "v": 1,
        "id": "ckpt-1",
        "ts": "2026-01-01T00:00:00Z",
        "channel_values": {"messages": []},
        "channel_versions": {"messages": 1},
        "versions_seen": {"__start__": {"messages": 0}},
        "updated_channels": ["messages"],
    }


class TestCheckpointGuards:
    def test_accepts_valid_checkpoint(self) -> None:
        assert _is_checkpoint(_valid_checkpoint())

    def test_rejects_missing_required_fields(self) -> None:
        payload = _valid_checkpoint()
        del payload["v"]
        assert not _is_checkpoint(payload)

    def test_rejects_invalid_channel_versions_scalar(self) -> None:
        payload = _valid_checkpoint()
        payload["channel_versions"] = {"messages": {"bad": "nested"}}
        assert not _is_checkpoint(payload)

    def test_rejects_invalid_versions_seen(self) -> None:
        payload = _valid_checkpoint()
        payload["versions_seen"] = {"__start__": "not-a-map"}
        assert not _is_checkpoint(payload)

    def test_rejects_invalid_updated_channels(self) -> None:
        payload = _valid_checkpoint()
        payload["updated_channels"] = [1, 2]
        assert not _is_checkpoint(payload)

    def test_accepts_checkpoint_without_updated_channels(self) -> None:
        payload = _valid_checkpoint()
        del payload["updated_channels"]
        assert _is_checkpoint(payload)


class TestCheckpointMetadataGuards:
    def test_accepts_valid_metadata(self) -> None:
        assert _is_checkpoint_metadata(
            {"source": "loop", "step": 3, "run_id": "run-1", "parents": {"a": "b"}}
        )

    def test_accepts_empty_metadata_dict(self) -> None:
        assert _is_checkpoint_metadata({})

    def test_rejects_invalid_source(self) -> None:
        assert not _is_checkpoint_metadata({"source": "unknown"})

    def test_rejects_invalid_parents(self) -> None:
        assert not _is_checkpoint_metadata({"parents": {"a": 1}})


class TestPendingWriteGuard:
    def test_accepts_valid_pending_write(self) -> None:
        assert _is_pending_write(("task-1", "messages", {"x": 1}))


class TestSpanDictGuard:
    def test_accepts_valid_span_tree(self) -> None:
        span = {
            "node": "agent",
            "type": "llm",
            "ignore": False,
            "children": [{"tool": "search", "type": "tool"}],
        }
        assert is_span_dict(span)

    def test_rejects_wrong_field_types(self) -> None:
        assert not is_span_dict({"node": 42})
        assert not is_span_dict({"children": [{"node": 1}]})

    def test_rejects_unrecognized_empty_dict(self) -> None:
        assert not is_span_dict({})
