"""gRPC server for Router Service."""

from __future__ import annotations

import asyncio

import grpc
from contextunity.core import (
    get_contextunit_logger,
    load_shared_config_from_env,
    router_pb2_grpc,
    setup_logging,
)

from .dispatcher_service import DispatcherService

logger = get_contextunit_logger(__name__)


async def serve():
    """Start the gRPC server for Router Service. Config: .env loaded only via Config.load() (single entry)."""
    from contextunity.router.core import set_core_config
    from contextunity.router.core.config import Config

    cfg = Config.load()  # Load service .env once
    set_core_config(cfg)

    config = load_shared_config_from_env()
    setup_logging(config=config, service_name="contextunity.router")

    # Silence chatty third-party loggers
    import logging as _logging

    get_contextunit_logger("httpx").setLevel(_logging.WARNING)

    # Build interceptor list: security + domain permission checks
    from .interceptors import RouterPermissionInterceptor

    interceptors = []
    interceptors.append(
        RouterPermissionInterceptor(
            shield_url=cfg.router.shield_grpc_host or config.shield_url,
        )
    )

    server = grpc.aio.server(
        interceptors=interceptors,
        options=(
            ("grpc.so_reuseport", 1 if config.grpc_reuse_port else 0),
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ),
    )

    # Register Dispatcher Service
    dispatcher = DispatcherService()
    router_pb2_grpc.add_RouterServiceServicer_to_server(dispatcher, server)
    logger.info("Dispatcher Service registered")

    port = cfg.router.port

    from contextunity.core.grpc_utils import graceful_shutdown, start_grpc_server

    heartbeat_task = await start_grpc_server(
        server,
        "router",
        port,
        instance_name=cfg.router.instance_name,
        tenants=cfg.router.tenants,
    )

    # Restore persisted project registrations from Redis
    await dispatcher.restore_registrations()

    # Wait for shutdown signal, drain streams, stop server
    from contextunity.router.service.stream_executors import get_stream_executor_manager

    await graceful_shutdown(
        server,
        "Router",
        heartbeat_task=heartbeat_task,
        before_stop=get_stream_executor_manager().drain_all,
    )


if __name__ == "__main__":
    # .env loaded in Config.load() when router runs; no load_dotenv here (single config entry)
    asyncio.run(serve())
