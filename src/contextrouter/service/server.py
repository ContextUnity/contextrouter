"""gRPC server for Router Service."""

from __future__ import annotations

import asyncio
import signal

import grpc
from contextcore import (
    get_context_unit_logger,
    load_shared_config_from_env,
    register_service,
    router_pb2_grpc,
    setup_logging,
)
from contextcore.security import get_security_interceptors, shield_status

from .dispatcher_service import DispatcherService

logger = get_context_unit_logger(__name__)


async def serve():
    """Start the gRPC server for Router Service. Config: .env loaded only via Config.load() (single entry)."""
    from contextrouter.core import set_core_config
    from contextrouter.core.config import Config

    cfg = Config.load()  # Load service .env once
    set_core_config(cfg)

    config = load_shared_config_from_env()
    setup_logging(config=config, service_name="contextrouter")

    # Silence chatty third-party loggers
    import logging as _logging

    _logging.getLogger("httpx").setLevel(_logging.WARNING)

    # Build interceptor list: security + domain permission checks
    from .interceptors import RouterPermissionInterceptor

    interceptors = list(get_security_interceptors())
    interceptors.append(RouterPermissionInterceptor())

    server = grpc.aio.server(interceptors=interceptors)

    # Log security status
    sec = shield_status()
    sec_log = logger.info if sec["security_enabled"] else logger.warning
    sec_log(
        "Security: enabled=%s, shield=%s",
        sec["security_enabled"],
        "active" if sec["shield_active"] else "not installed",
    )

    # Register Dispatcher Service
    dispatcher = DispatcherService()
    router_pb2_grpc.add_RouterServiceServicer_to_server(dispatcher, server)
    logger.info("Dispatcher Service registered")

    port = cfg.router.port
    instance_name = cfg.router.instance_name

    from contextcore.grpc_utils import create_server_credentials

    tls_creds = create_server_credentials()
    if tls_creds:
        server.add_secure_port(f"[::]:{port}", tls_creds)
        logger.info("Router Service starting on :%s with TLS (instance=%s)", port, instance_name)
    else:
        server.add_insecure_port(f"[::]:{port}")
        logger.info("Router Service starting on :%s (instance=%s)", port, instance_name)
    await server.start()

    # Restore persisted project registrations from Redis
    await dispatcher.restore_registrations()

    # Register in Redis for service discovery
    tenants = cfg.router.tenants

    heartbeat_task = await register_service(
        service="router",
        instance=instance_name,
        endpoint=f"localhost:{port}",
        tenants=tenants,
        metadata={"port": int(port)},
    )

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown_handler():
        logger.info("Shutdown signal received, stopping Router...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown_handler)

    await stop_event.wait()
    logger.info("Stopping gRPC server (5s grace)...")
    await server.stop(grace=5)
    if heartbeat_task:
        heartbeat_task.cancel()
    logger.info("Router server stopped.")


if __name__ == "__main__":
    # .env loaded in Config.load() when router runs; no load_dotenv here (single config entry)
    asyncio.run(serve())
