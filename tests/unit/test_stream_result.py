"""Unit tests for federated stream result validation boundaries."""

from __future__ import annotations

from datetime import datetime

import pytest

from contextunity.router.service.stream_result import validate_stream_result


def test_validate_stream_result_accepts_json_object() -> None:
    wire = validate_stream_result({"rows": [{"id": 1, "ok": True}]})
    assert wire["rows"][0]["id"] == 1


def test_validate_stream_result_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match="Stream result must be a JSON object"):
        validate_stream_result({"rows": [{"created_at": datetime(2024, 1, 1)}]})


def test_validate_stream_result_rejects_excess_keys() -> None:
    payload = {f"k{i}": i for i in range(300)}
    with pytest.raises(ValueError, match="maximum key count"):
        validate_stream_result(payload)


def test_validate_stream_result_rejects_deep_nesting() -> None:
    nested: dict[str, object] = {"v": 1}
    current: dict[str, object] = nested
    for _ in range(12):
        inner: dict[str, object] = {"v": current}
        current = inner
    with pytest.raises(ValueError, match="maximum nesting depth"):
        validate_stream_result({"tree": current})
