"""Typed Redis persistence for RegisterManifest recovery.

Single choke point for Router registration storage: isolates redis-py ``Any``
and stdlib JSON from ``PersistenceMixin`` business logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from typing import Protocol

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps, json_loads
from contextunity.core.types import JsonDict, is_json_dict

logger = get_contextunit_logger(__name__)

_DEFAULT_PREFIX = "router:registrations"


def _parse_registration_payload(raw: str) -> JsonDict | None:
    """Parse persisted registration JSON through the core L1→L2 boundary."""
    decoded = json_loads(raw)
    if is_json_dict(decoded):
        return decoded
    return None


class RegistrationRedisOps(Protocol):
    """Async Redis operations required for registration persistence."""

    async def set(self, key: str, value: str) -> None: ...

    async def get(self, key: str) -> str | None: ...

    async def delete(self, *keys: str) -> int: ...

    def scan_iter(self, match: str) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


class _AsyncRedisSet(Protocol):
    def set(self, name: str, value: str) -> Awaitable[object]: ...


class _AsyncRedisGet(Protocol):
    def get(self, name: str) -> Awaitable[object]: ...


class _AsyncRedisDelete(Protocol):
    def delete(self, *names: str) -> Awaitable[object]: ...


class _AsyncRedisScan(Protocol):
    def scan_iter(
        self,
        match: str | None = None,
        count: int | None = None,
        _type: str | None = None,
    ) -> AsyncIterator[object]: ...


class _AsyncRedisClose(Protocol):
    def aclose(self) -> Awaitable[object]: ...


def _async_redis_set(client: _AsyncRedisSet, name: str, value: str) -> Awaitable[object]:
    return client.set(name, value)


def _async_redis_get(client: _AsyncRedisGet, name: str) -> Awaitable[object]:
    return client.get(name)


def _async_redis_delete(client: _AsyncRedisDelete, *names: str) -> Awaitable[object]:
    return client.delete(*names)


def _async_redis_close(client: _AsyncRedisClose) -> Awaitable[object]:
    return client.aclose()


async def _async_redis_scan_keys(
    client: _AsyncRedisScan,
    pattern: str,
) -> AsyncIterator[str]:
    async for raw in client.scan_iter(match=pattern):
        if isinstance(raw, str):
            yield raw


class AsyncRegistrationRedisClient:
    """Typed async Redis client for registration persistence.

    Mirrors ``contextunity.core.discovery.client.SyncRedisClient``: narrows
    redis-py ``Any`` return types at the adapter boundary.
    """

    def __init__(self, url: str) -> None:
        import redis.asyncio as _redis

        self._r: _redis.Redis = _redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

    async def set(self, key: str, value: str) -> None:
        _ = await _async_redis_set(self._r, key, value)

    async def get(self, key: str) -> str | None:
        val: object = await _async_redis_get(self._r, key)
        if val is None:
            return None
        return val if isinstance(val, str) else str(val)

    async def delete(self, *keys: str) -> int:
        val: object = await _async_redis_delete(self._r, *keys)
        return val if isinstance(val, int) else 0

    async def scan_iter(self, match: str) -> AsyncIterator[str]:
        async for key in _async_redis_scan_keys(self._r, match):
            yield key

    async def aclose(self) -> None:
        _ = await _async_redis_close(self._r)


class RegistrationRedisStore:
    """Registration bundle CRUD against Redis with typed wire parsing."""

    _client: RegistrationRedisOps
    _prefix: str

    def __init__(
        self,
        client: RegistrationRedisOps,
        *,
        prefix: str = _DEFAULT_PREFIX,
    ) -> None:
        self._client = client
        self._prefix = prefix

    def project_key(self, project_id: str) -> str:
        return f"{self._prefix}:{project_id}"

    def _project_key(self, project_id: str) -> str:
        return self.project_key(project_id)

    def _hash_key(self, project_id: str) -> str:
        return f"{self._prefix}:{project_id}:hash"

    def _stream_key(self, project_id: str) -> str:
        return f"{self._prefix}:{project_id}:stream"

    async def write_stream_secret(self, project_id: str, stream_secret: str) -> None:
        """Persist BiDi stream auth secret under the registration Redis namespace."""
        await self._client.set(self._stream_key(project_id), stream_secret)

    async def read_stream_secret(self, project_id: str) -> str | None:
        """Read persisted BiDi stream auth secret, if any."""
        return await self._client.get(self._stream_key(project_id))

    async def persist(self, project_id: str, payload: dict[str, object]) -> None:
        """Serialize and store a registration payload."""
        await self._client.set(
            self._project_key(project_id),
            json_dumps(payload, default=str),
        )

    async def remove(self, project_id: str) -> int:
        """Delete persisted payload, manifest hash, and stream secret keys."""
        return await self._client.delete(
            self._project_key(project_id),
            self._hash_key(project_id),
            self._stream_key(project_id),
        )

    async def read_hash(self, project_id: str) -> str | None:
        """Return stored manifest hash, if any."""
        return await self._client.get(self._hash_key(project_id))

    async def write_hash(self, project_id: str, manifest_hash: str) -> None:
        """Persist manifest hash for idempotent RegisterManifest."""
        await self._client.set(self._hash_key(project_id), manifest_hash)

    async def list_registration_keys(self) -> list[str]:
        """List all registration keys (hash filtering is caller-side)."""
        keys: list[str] = []
        async for key in self._client.scan_iter(f"{self._prefix}:*"):
            keys.append(key)
        return keys

    async def load_payload(self, key: str) -> JsonDict | None:
        """Load and parse a registration payload from a full Redis key."""
        raw = await self._client.get(key)
        if not raw:
            return None
        return _parse_registration_payload(raw)

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._client.aclose()


async def open_registration_redis_store() -> RegistrationRedisStore | None:
    """Open a registration store from Router shared config, or ``None`` if disabled."""
    try:
        from contextunity.router.core import get_core_config

        config = get_core_config()
        if not config.redis.enabled or not config.redis.url:
            return None

        client = AsyncRegistrationRedisClient(config.redis.url)
        return RegistrationRedisStore(client)
    except Exception as exc:  # graceful-degrade: Redis optional
        logger.warning("Redis not available for registration persistence: %s", exc)
        return None


__all__ = [
    "AsyncRegistrationRedisClient",
    "RegistrationRedisOps",
    "RegistrationRedisStore",
    "open_registration_redis_store",
]
