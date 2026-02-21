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
        from contextrouter.service.security import sanitize_for_struct

        assert sanitize_for_struct(None) is None

    def test_bool_passthrough(self):
        from contextrouter.service.security import sanitize_for_struct

        assert sanitize_for_struct(True) is True

    def test_int_passthrough(self):
        from contextrouter.service.security import sanitize_for_struct

        assert sanitize_for_struct(42) == 42

    def test_float_passthrough(self):
        from contextrouter.service.security import sanitize_for_struct

        assert sanitize_for_struct(3.14) == pytest.approx(3.14)

    def test_string_passthrough(self):
        from contextrouter.service.security import sanitize_for_struct

        assert sanitize_for_struct("hello") == "hello"

    def test_dict_recursion(self):
        from contextrouter.service.security import sanitize_for_struct

        result = sanitize_for_struct({"key": "value", "nested": {"a": 1}})
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_list_recursion(self):
        from contextrouter.service.security import sanitize_for_struct

        result = sanitize_for_struct([1, "two", None])
        assert result == [1, "two", None]

    def test_tuple_converted_to_list(self):
        from contextrouter.service.security import sanitize_for_struct

        result = sanitize_for_struct((1, 2, 3))
        assert result == [1, 2, 3]

    def test_uuid_converted_to_string(self):
        from contextrouter.service.security import sanitize_for_struct

        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = sanitize_for_struct(u)
        assert isinstance(result, str)
        assert "12345678" in result

    def test_pydantic_model_converted(self):
        from contextrouter.service.security import sanitize_for_struct

        model = MagicMock()
        model.model_dump.return_value = {"field": "value"}
        result = sanitize_for_struct(model)
        assert result == {"field": "value"}

    def test_dict_with_none_values(self):
        from contextrouter.service.security import sanitize_for_struct

        result = sanitize_for_struct({"a": None, "b": "ok", "c": None})
        assert result == {"a": None, "b": "ok", "c": None}

    def test_nested_complex_structure(self):
        from contextrouter.service.security import sanitize_for_struct

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
# Decorator exports
# ---------------------------------------------------------------------------


class TestDecoratorExports:
    """Ensure decorator module exports are correct."""

    def test_grpc_error_handler_importable(self):
        from contextrouter.service.decorators import grpc_error_handler

        assert callable(grpc_error_handler)

    def test_grpc_stream_error_handler_importable(self):
        from contextrouter.service.decorators import grpc_stream_error_handler

        assert callable(grpc_stream_error_handler)

    def test_decorator_all_exports(self):
        from contextrouter.service.decorators import __all__

        assert "grpc_error_handler" in __all__
        assert "grpc_stream_error_handler" in __all__


# ---------------------------------------------------------------------------
# Security module exports
# ---------------------------------------------------------------------------


class TestSecurityExports:
    """Ensure security module exports are correct."""

    def test_security_all_exports(self):
        from contextrouter.service.security import __all__

        assert "sanitize_for_struct" in __all__
        assert "validate_dispatcher_access" in __all__


# ---------------------------------------------------------------------------
# Mixin exports
# ---------------------------------------------------------------------------


class TestMixinExports:
    """Ensure mixin modules are properly importable."""

    def test_execution_mixin_importable(self):
        from contextrouter.service.mixins import ExecutionMixin

        assert ExecutionMixin is not None

    def test_registration_mixin_importable(self):
        from contextrouter.service.mixins import RegistrationMixin

        assert RegistrationMixin is not None

    def test_persistence_mixin_importable(self):
        from contextrouter.service.mixins import PersistenceMixin

        assert PersistenceMixin is not None

    def test_dispatcher_service_uses_all_mixins(self):
        from contextrouter.service.dispatcher_service import DispatcherService
        from contextrouter.service.mixins import ExecutionMixin, PersistenceMixin, RegistrationMixin

        assert issubclass(DispatcherService, ExecutionMixin)
        assert issubclass(DispatcherService, RegistrationMixin)
        assert issubclass(DispatcherService, PersistenceMixin)
