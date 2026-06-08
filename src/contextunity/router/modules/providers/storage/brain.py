"""Brain provider (calls contextunity.brain service via gRPC).
This provider delegates retrieval to the centralized contextunity.brain service.
Uses ContextUnit as the universal data contract for all operations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

import grpc
from contextunity.core import (
    ContextUnit,
    brain_pb2_grpc,
    contextunit_pb2,
    get_contextunit_logger,
)
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.permissions import Permissions
from contextunity.core.sdk.payload import get_str
from contextunity.core.types import JsonDict

from contextunity.router.core.config import get_core_config
from contextunity.router.core.interfaces import BaseProvider, IRead
from contextunity.router.modules.providers._async_iterate import async_iterate

logger = get_contextunit_logger(__name__)


async def _brain_search_stream(
    stub: object,
    request: object,
) -> AsyncIterator[contextunit_pb2.ContextUnit]:
    search_fn: object = getattr(stub, "Search", None)
    if not callable(search_fn):
        raise ConfigurationError("Brain gRPC stub is missing Search")
    stream_obj: object = search_fn(request)
    async for response in async_iterate(stream_obj):
        if isinstance(response, contextunit_pb2.ContextUnit):
            yield response


async def _brain_upsert(
    stub: object,
    request: object,
) -> contextunit_pb2.ContextUnit:
    upsert_fn: object = getattr(stub, "Upsert", None)
    if not callable(upsert_fn):
        raise ConfigurationError("Brain gRPC stub is missing Upsert")
    upsert_call: object = upsert_fn(request)
    from contextunity.core.narrowing import await_object

    resolved: object = await await_object(upsert_call)
    if not isinstance(resolved, contextunit_pb2.ContextUnit):
        raise ConfigurationError("Brain Upsert returned unexpected response type")
    return resolved


class BrainProvider(BaseProvider, IRead):
    """Provider that delegates to contextunity.brain service using ContextUnit protocol."""

    endpoint: str
    _channel: grpc.aio.Channel
    _stub: object
    Permissions: type[Permissions]

    def __init__(self, **kwargs: object) -> None:
        """Resolve Brain gRPC endpoint from ``SharedConfig`` and create the channel/stub."""
        _ = kwargs
        config = get_core_config()

        # Use RouterConfig brain_url (resolved from router.yml/env/discovery)
        self.endpoint = config.brain_url

        if self.endpoint and self.endpoint.startswith("postgres"):
            raise ConfigurationError(
                (
                    f"Invalid Brain gRPC endpoint: expected a host:port for gRPC, but got a database URL "
                    f"('{self.endpoint}'). Check CU_BRAIN_GRPC_URL in your environment "
                    "or setup discovery via Redis."
                )
            )

        logger.info("Initializing BrainProvider (endpoint: %s)", self.endpoint)
        from contextunity.core.grpc_utils import create_channel

        self._channel = create_channel(self.endpoint, config=config)
        self._stub = brain_pb2_grpc.BrainServiceStub(self._channel)
        self.Permissions = Permissions

    @override
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: JsonDict | None = None,
    ) -> list[ContextUnit]:
        """Retrieve from brain using contextunit protocol."""
        # Build ContextUnit request
        unit = ContextUnit(
            payload={
                "tenant_id": filters.get("tenant_id", "default") if filters else "default",
                "query_text": query,
                "limit": limit,
                "source_types": filters.get("source_types", []) if filters else [],
            },
            provenance=["router:brain_provider:read"],
        )

        units: list[ContextUnit] = []
        try:
            req = unit.to_protobuf(contextunit_pb2)
            async for response in _brain_search_stream(self._stub, req):
                result = ContextUnit.from_protobuf(response)
                units.append(result)
                if len(units) >= limit:
                    break
        except grpc.RpcError as e:
            logger.error("Brain gRPC call failed: %s", e)
            raise

        return units

    async def upsert(
        self,
        content: str,
        *,
        tenant_id: str = "default",
        source_type: str = "document",
        metadata: JsonDict | None = None,
    ) -> str:
        """Upsert content to brain using contextunit protocol."""
        unit = ContextUnit(
            payload={
                "tenant_id": tenant_id,
                "content": content,
                "source_type": source_type,
                "metadata": metadata or {},
            },
            provenance=["router:brain_provider:upsert"],
        )

        try:
            req = unit.to_protobuf(contextunit_pb2)
            response = await _brain_upsert(self._stub, req)
            result = ContextUnit.from_protobuf(response)
            return get_str(result.payload, "id")
        except grpc.RpcError as e:
            logger.error("Brain gRPC upsert failed: %s", e)
            raise

    @override
    async def sink(self, unit: ContextUnit) -> None:
        """Sink delegates to brain upsert via protobuf stub."""
        try:
            req = unit.to_protobuf(contextunit_pb2)
            _ = await _brain_upsert(self._stub, req)
        except grpc.RpcError as e:
            logger.error("Brain gRPC sink failed: %s", e)
            raise

    async def close(self):
        """Clean up resources."""
        if hasattr(self, "_channel") and self._channel:
            await self._channel.close()


__all__ = ["BrainProvider"]
