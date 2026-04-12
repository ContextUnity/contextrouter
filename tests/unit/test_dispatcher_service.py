"""Unit tests for dispatcher service security and decorator modules."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sanitize_for_struct
# ---------------------------------------------------------------------------


class TestSanitizeForStruct:
    """Tests for protobuf Struct-safe value sanitization."""

    def test_none_passthrough(self):
        from contextunity.router.service.security import sanitize_for_struct

        assert sanitize_for_struct(None) is None

    def test_bool_passthrough(self):
        from contextunity.router.service.security import sanitize_for_struct

        assert sanitize_for_struct(True) is True

    def test_int_passthrough(self):
        from contextunity.router.service.security import sanitize_for_struct

        assert sanitize_for_struct(42) == 42

    def test_float_passthrough(self):
        from contextunity.router.service.security import sanitize_for_struct

        assert sanitize_for_struct(3.14) == pytest.approx(3.14)

    def test_string_passthrough(self):
        from contextunity.router.service.security import sanitize_for_struct

        assert sanitize_for_struct("hello") == "hello"

    def test_dict_recursion(self):
        from contextunity.router.service.security import sanitize_for_struct

        result = sanitize_for_struct({"key": "value", "nested": {"a": 1}})
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_list_recursion(self):
        from contextunity.router.service.security import sanitize_for_struct

        result = sanitize_for_struct([1, "two", None])
        assert result == [1, "two", None]

    def test_tuple_converted_to_list(self):
        from contextunity.router.service.security import sanitize_for_struct

        result = sanitize_for_struct((1, 2, 3))
        assert result == [1, 2, 3]

    def test_uuid_converted_to_string(self):
        from contextunity.router.service.security import sanitize_for_struct

        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = sanitize_for_struct(u)
        assert isinstance(result, str)
        assert "12345678" in result

    def test_pydantic_model_converted(self):
        from contextunity.router.service.security import sanitize_for_struct

        model = MagicMock()
        model.model_dump.return_value = {"field": "value"}
        result = sanitize_for_struct(model)
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


class TestDispatcherServiceCore:
    """Ensure DispatcherService initializes correctly."""

    def test_dispatcher_service_initialization(self):
        from contextunity.router.service.dispatcher_service import DispatcherService

        service = DispatcherService()
        assert isinstance(service._project_tools, dict)
        assert isinstance(service._project_configs, dict)
        assert isinstance(service._stream_secrets, dict)
        assert not service._project_tools
        assert not service._project_configs
