"""Payload construction and config extraction for Router gRPC service handlers."""

from __future__ import annotations

from contextunity.core.sdk.service_helpers import (
    contextunit_error_response_factory,
    make_response,
    parse_unit,
)

router_error_response_factory = contextunit_error_response_factory

__all__ = ["parse_unit", "make_response", "router_error_response_factory"]
