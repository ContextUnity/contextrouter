"""Example gRPC client for Dispatcher Agent.

This example demonstrates how to use the dispatcher agent via gRPC
with ContextUnit protocol and SecurityScopes.
"""

from __future__ import annotations

import asyncio

import grpc
from contextcore import ContextUnit, SecurityScopes, create_channel, router_pb2_grpc


async def example_execute_dispatcher() -> None:
    """Example: Execute dispatcher agent via gRPC."""
    print("=== Example: Execute Dispatcher via gRPC ===\n")

    # Create gRPC channel
    channel = create_channel("localhost:50050")
    stub = router_pb2_grpc.RouterServiceStub(channel)

    # Create ContextUnit with SecurityScopes
    unit = ContextUnit(
        payload={
            "tenant_id": "example_tenant",
            "messages": [{"role": "user", "content": "What tools are available in the system?"}],
            "session_id": "grpc_session_123",
            "platform": "grpc",
            "max_iterations": 10,
        },
        security=SecurityScopes(
            read=["dispatcher:execute"],  # Required scope
        ),
    )

    # Call gRPC method
    response_pb = await stub.ExecuteDispatcher(unit.to_protobuf())

    # Parse response
    response_unit = ContextUnit.from_protobuf(response_pb)
    print("Response:")
    print(f"  Trace ID: {response_unit.trace_id}")
    print(f"  Provenance: {response_unit.provenance}")
    print(f"  Payload: {response_unit.payload}")
    print()

    await channel.close()


async def example_stream_dispatcher() -> None:
    """Example: Stream dispatcher agent via gRPC."""
    print("=== Example: Stream Dispatcher via gRPC ===\n")

    # Create gRPC channel
    channel = create_channel("localhost:50050")
    stub = router_pb2_grpc.RouterServiceStub(channel)

    # Create ContextUnit
    unit = ContextUnit(
        payload={
            "tenant_id": "example_tenant",
            "messages": [
                {"role": "user", "content": "Search for information about Python async programming"}
            ],
            "session_id": "grpc_stream_session",
            "platform": "grpc",
        },
        security=SecurityScopes(
            read=["dispatcher:execute"],
        ),
    )

    # Stream events
    print("Streaming events:")
    async for event_pb in stub.StreamDispatcher(unit.to_protobuf()):
        event_unit = ContextUnit.from_protobuf(event_pb)
        print(f"  Event: {event_unit.payload}")

    print()
    await channel.close()


async def example_without_permission() -> None:
    """Example: What happens without required permission."""
    print("=== Example: Missing Permission ===\n")

    channel = create_channel("localhost:50050")
    stub = router_pb2_grpc.RouterServiceStub(channel)

    # Create ContextUnit WITHOUT required scope
    unit = ContextUnit(
        payload={
            "tenant_id": "example_tenant",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        security=SecurityScopes(
            read=["other:scope"],  # Missing dispatcher:execute
        ),
    )

    try:
        response_pb = await stub.ExecuteDispatcher(unit.to_protobuf())
        response_unit = ContextUnit.from_protobuf(response_pb)
        if "error" in response_unit.payload:
            print(f"Error (expected): {response_unit.payload['error']}")
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

    print()
    await channel.close()


async def main() -> None:
    """Run all examples."""
    try:
        await example_execute_dispatcher()
        await example_stream_dispatcher()
        await example_without_permission()
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")
        print("\nMake sure the gRPC server is running:")
        print("  python -m contextrouter.service")


if __name__ == "__main__":
    asyncio.run(main())
