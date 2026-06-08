"""Unit tests for dispatcher service security and decorator modules."""

from __future__ import annotations

import uuid

import pytest

# ---------------------------------------------------------------------------
# sanitize_for_struct
# ---------------------------------------------------------------------------


class TestSanitizeForStruct:
    """Tests for protobuf Struct-safe value sanitization."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (True, True),
            (42, 42),
            ("hello", "hello"),
        ],
    )
    def test_scalar_passthrough(self, value, expected):
        from contextunity.router.service.security import sanitize_for_struct

        assert sanitize_for_struct(value) == expected

    def test_dict_recursion(self):
        from contextunity.router.service.security import sanitize_for_struct

        result = sanitize_for_struct({"key": "value", "nested": {"a": 1}})
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_list_recursion(self):
        from contextunity.router.service.security import sanitize_for_struct

        result = sanitize_for_struct([1, "two", None])
        assert result == [1, "two", None]

    def test_uuid_converted_to_string(self):
        from contextunity.router.service.security import sanitize_for_struct

        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = sanitize_for_struct(u)
        assert isinstance(result, str)
        assert "12345678" in result

    def test_pydantic_model_converted(self):
        from contextunity.router.service.security import sanitize_for_struct

        class FakeModel:
            def model_dump(self):
                return {"field": "value"}

        result = sanitize_for_struct(FakeModel())
        assert result == {"field": "value"}

    def test_dict_with_none_values(self):
        from contextunity.router.service.security import sanitize_for_struct

        result = sanitize_for_struct({"a": None, "b": "ok", "c": None})
        assert result == {"a": None, "b": "ok", "c": None}

    def test_nested_complex_structure(self):
        from contextunity.router.service.security import sanitize_for_struct

        data = {
            "steps": [
                {"id": uuid.uuid4(), "data": {"nested": True}},
                {"id": uuid.uuid4(), "data": None},
            ],
            "count": 2,
        }
        result = sanitize_for_struct(data)
        assert isinstance(result["steps"], list)
        assert isinstance(result["steps"][0]["id"], str)
        assert result["steps"][0]["data"] == {"nested": True}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DispatcherService Core functionality
# ---------------------------------------------------------------------------
