"""Local factory for ContextRouter gracefully degraded execution."""

import grpc
from contextunity.core import get_contextunit_logger, router_pb2_grpc

from .dispatcher_service import DispatcherService

logger = get_contextunit_logger(__name__)


async def create_local_router() -> grpc.aio.Server:
    """Create a gracefully degraded local Router service."""
    logger.info("Initializing Local Router Service (In-Memory)")

    from contextunity.router.core.config import get_core_config as get_router_config

    from .interceptors import RouterPermissionInterceptor

    config = get_router_config()
    shield_url = config.shield_url
    logger.info("Local Router: shield_url=%s", shield_url or "(disabled)")

    server = grpc.aio.server(interceptors=[RouterPermissionInterceptor(shield_url=shield_url)])
    dispatcher = DispatcherService()

    router_pb2_grpc.add_RouterServiceServicer_to_server(dispatcher, server)
    _ = server.add_insecure_port(f"[::]:{config.port}")

    return server


if __name__ == "__main__":
    import asyncio

    from contextunity.core.logging import setup_logging

    setup_logging()

    async def _run() -> None:
        server = await create_local_router()
        _ = await server.start()
        print("Router gRPC listening (local)")
        _ = await server.wait_for_termination()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
