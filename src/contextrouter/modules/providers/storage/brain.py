"""Brain provider (calls ContextBrain service).

This provider delegates retrieval to the centralized ContextBrain service.
Supports both "local" (direct library call) and "grpc" (network call) modes.

Uses ContextUnit as the universal data contract for all operations.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc
from contextcore import ContextToken, ContextUnit, brain_pb2_grpc, context_unit_pb2

from contextrouter.core.config import get_core_config
from contextrouter.core.interfaces import BaseProvider, IRead, secured

logger = logging.getLogger(__name__)


class BrainProvider(BaseProvider, IRead):
    """Provider that delegates to ContextBrain service using ContextUnit protocol."""

    def __init__(self, **kwargs: Any) -> None:
        cfg = get_core_config()
        self.mode = cfg.brain.mode
        self.endpoint = cfg.brain.grpc_endpoint

        if self.mode == "local":
            logger.info("Initializing BrainProvider in LOCAL mode")
            from contextbrain import BrainService

            self.service = BrainService()
            self._stub = None
        else:
            logger.info("Initializing BrainProvider in GRPC mode (endpoint: %s)", self.endpoint)
            self.service = None
            # Channel is usually managed externally or kept open
            self._channel = grpc.aio.insecure_channel(self.endpoint)
            self._stub = brain_pb2_grpc.BrainServiceStub(self._channel)

    @secured()
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: ContextToken,
    ) -> list[ContextUnit]:
        """Retrieve from Brain using ContextUnit protocol."""
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

        units = []
        if self.mode == "local":
            # Direct library call
            req = unit.to_protobuf(context_unit_pb2)
            async for response in self.service.Search(req, context=None):
                result = ContextUnit.from_protobuf(response)
                units.append(result)
                if len(units) >= limit:
                    break
        else:
            # gRPC remote call
            try:
                req = unit.to_protobuf(context_unit_pb2)
                stream = self._stub.Search(req)
                async for response in stream:
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
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upsert content to Brain using ContextUnit protocol."""
        unit = ContextUnit(
            payload={
                "tenant_id": tenant_id,
                "content": content,
                "source_type": source_type,
                "metadata": metadata or {},
            },
            provenance=["router:brain_provider:upsert"],
        )

        if self.mode == "local":
            req = unit.to_protobuf(context_unit_pb2)
            response = await self.service.Upsert(req, context=None)
            result = ContextUnit.from_protobuf(response)
            return result.payload.get("id", "")
        else:
            try:
                req = unit.to_protobuf(context_unit_pb2)
                response = await self._stub.Upsert(req)
                result = ContextUnit.from_protobuf(response)
                return result.payload.get("id", "")
            except grpc.RpcError as e:
                logger.error("Brain gRPC upsert failed: %s", e)
                raise

    async def close(self):
        """Clean up resources."""
        if hasattr(self, "_channel") and self._channel:
            await self._channel.close()


__all__ = ["BrainProvider"]
