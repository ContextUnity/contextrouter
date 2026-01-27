"""Brain provider (calls ContextBrain options).

This provider delegates retrieval to the centralized ContextBrain service.
Supports both "local" (direct library call) and "grpc" (network call) modes.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc
from contextcore import ContextToken, ContextUnit, brain_pb2, brain_pb2_grpc

from contextrouter.core.config import get_core_config
from contextrouter.core.interfaces import BaseProvider, IRead, secured

logger = logging.getLogger(__name__)


class BrainProvider(BaseProvider, IRead):
    """Provider that delegates to ContextBrain service."""

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
        """Retrieve from Brain."""
        # Construct request
        # NOTE: Using Struct for payload as defined in proto
        from google.protobuf.struct_pb2 import Struct

        payload_struct = Struct()
        payload_dict = {"content": query}
        if filters:
            payload_dict["filters"] = filters
        payload_struct.update(payload_dict)

        req = brain_pb2.ContextUnit(
            payload=payload_struct,
            modality=0,  # TEXT
        )

        units = []
        if self.mode == "local":
            # Direct library call
            async for unit in self.service.QueryMemory(req, context=None):
                units.append(unit)
                if len(units) >= limit:
                    break
        else:
            # gRPC remote call
            try:
                # QueryMemory returns a stream
                stream = self._stub.QueryMemory(req)
                async for unit in stream:
                    units.append(unit)
                    if len(units) >= limit:
                        break
            except grpc.RpcError as e:
                logger.error("Brain gRPC call failed: %s", e)
                raise

        return units

    async def close(self):
        """Clean up resources."""
        if hasattr(self, "_channel") and self._channel:
            await self._channel.close()


__all__ = ["BrainProvider"]
