"""Integration: RegisterManifest Redis persistence ↔ projection ↔ in-memory restore.

Exercises the typed ``RegistrationRedisStore`` choke point and
``registered_project_config_from_persisted`` without a live Redis server.

``project_id`` and ``allowed_tenants`` are intentionally different strings to
guard against conflating registration identity with execution tenant scope.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from contextunity.core.parsing import json_dumps
from contextunity.core.types import is_json_dict

from contextunity.router.service.mixins.persistence import PersistenceMixin
from contextunity.router.service.registration_projection import (
    registered_project_config_from_persisted,
)
from contextunity.router.service.registration_redis import RegistrationRedisStore

_PROJECT_ID = "nszu-platform"
_TENANTS = ("clinical-tenant-a", "clinical-tenant-b")
_MANIFEST_HASH = "sha256:deadbeef"
_STREAM_SECRET = "stream-secret-roundtrip-001"


def _sample_registration_payload() -> dict[str, object]:
    return {
        "project_id": _PROJECT_ID,
        "allowed_tenants": list(_TENANTS),
        "default_graph": "analytics",
        "policy": {
            "models": {
                "llm": {"default": "openai/gpt-5-mini", "fallback": ["inception/mercury-2"]},
            },
            "langfuse": {"tracing_enabled": True},
        },
        "services": {"router": {"enabled": True}, "brain": {"enabled": True}},
        "tools": [
            {
                "name": "medical_sql",
                "type": "sql",
                "description": "Hospital analytics SQL",
                "config": {"timeout": 120},
            },
        ],
        "graph": {
            "analytics": {
                "nodes": [
                    {
                        "name": "planner",
                        "type": "llm",
                        "model": "openai/gpt-5-mini",
                        "pii_masking": True,
                    },
                    {
                        "name": "executor",
                        "type": "tool",
                        "tool_binding": "federated:medical_sql",
                    },
                ],
                "edges": [
                    {"from_node": "__start__", "to_node": "planner"},
                    {"from_node": "planner", "to_node": "executor"},
                ],
            },
        },
    }


class _InMemoryRedis:
    """Minimal async Redis double implementing ``RegistrationRedisOps``."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.deleted: list[tuple[str, ...]] = []

    async def set(self, key: str, value: str) -> None:
        self.data[key] = value

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def delete(self, *keys: str) -> int:
        self.deleted.append(keys)
        removed = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                removed += 1
        return removed

    async def scan_iter(self, match: str) -> AsyncIterator[str]:
        prefix = match.removesuffix("*")
        for key in sorted(self.data):
            if key.startswith(prefix):
                yield key

    async def aclose(self) -> None:
        return None


@pytest.fixture
def redis_backend() -> _InMemoryRedis:
    return _InMemoryRedis()


@pytest.fixture
def store(redis_backend: _InMemoryRedis) -> RegistrationRedisStore:
    return RegistrationRedisStore(redis_backend)


@pytest.mark.asyncio
async def test_registration_redis_payload_round_trip(store: RegistrationRedisStore) -> None:
    """``persist`` → ``load_payload`` preserves JSON registration bundle."""
    payload = _sample_registration_payload()
    await store.persist(_PROJECT_ID, payload)

    loaded = await store.load_payload(store.project_key(_PROJECT_ID))
    assert loaded is not None
    assert loaded.get("project_id") == _PROJECT_ID
    assert loaded.get("allowed_tenants") == list(_TENANTS)
    assert is_json_dict(loaded.get("graph"))

    raw = await store._client.get(store.project_key(_PROJECT_ID))
    assert raw is not None
    reparsed = json.loads(raw)
    assert reparsed["project_id"] == _PROJECT_ID
    assert reparsed["default_graph"] == "analytics"


@pytest.mark.asyncio
async def test_registration_redis_hash_and_stream_sidecars(
    store: RegistrationRedisStore,
    redis_backend: _InMemoryRedis,
) -> None:
    """Hash idempotency and BiDi stream secrets use dedicated keys."""
    await store.persist(_PROJECT_ID, _sample_registration_payload())
    await store.write_hash(_PROJECT_ID, _MANIFEST_HASH)
    await store.write_stream_secret(_PROJECT_ID, _STREAM_SECRET)

    assert await store.read_hash(_PROJECT_ID) == _MANIFEST_HASH
    assert await store.read_stream_secret(_PROJECT_ID) == _STREAM_SECRET

    keys = await store.list_registration_keys()
    assert f"router:registrations:{_PROJECT_ID}" in keys
    assert f"router:registrations:{_PROJECT_ID}:hash" in keys
    assert f"router:registrations:{_PROJECT_ID}:stream" in keys

    deleted = await store.remove(_PROJECT_ID)
    assert deleted == 3
    assert store.project_key(_PROJECT_ID) not in redis_backend.data
    assert await store.read_hash(_PROJECT_ID) is None
    assert await store.read_stream_secret(_PROJECT_ID) is None


@pytest.mark.asyncio
async def test_registration_projection_preserves_tenants_distinct_from_project(
    store: RegistrationRedisStore,
) -> None:
    """Projection rebuilds L4 config; tenants are not aliased to ``project_id``."""
    payload = _sample_registration_payload()
    await store.persist(_PROJECT_ID, payload)
    loaded = await store.load_payload(store.project_key(_PROJECT_ID))
    assert loaded is not None

    graph_map = loaded.get("graph")
    assert is_json_dict(graph_map)

    config = registered_project_config_from_persisted(dict(loaded), graph_map)
    assert config.get("project_id") == _PROJECT_ID
    assert config.get("allowed_tenants") == list(_TENANTS)
    assert config["allowed_tenants"][0] != _PROJECT_ID

    tools = config.get("tools")
    assert isinstance(tools, list) and len(tools) == 1
    assert tools[0]["name"] == "medical_sql"

    nodes = config.get("nodes")
    assert isinstance(nodes, list) and len(nodes) == 2
    node_names = {node["name"] for node in nodes}
    assert node_names == {"planner", "executor"}
    assert all(node.get("graph_key") == "analytics" for node in nodes)


@pytest.mark.asyncio
async def test_persistence_restore_round_trip_via_registration_store(
    redis_backend: _InMemoryRedis,
) -> None:
    """Simulate restart: Redis write → empty memory → ``_restore_project_from_persistence``."""
    store = RegistrationRedisStore(redis_backend)
    payload = _sample_registration_payload()
    await store.persist(_PROJECT_ID, payload)
    await store.write_hash(_PROJECT_ID, _MANIFEST_HASH)
    await store.close()

    class _Harness(PersistenceMixin):
        def __init__(self, backend: _InMemoryRedis) -> None:
            self._backend = backend
            self._project_tools: dict[str, list[str]] = {}
            self._project_graphs: dict[str, dict[str, str]] = {}
            self._project_router_callbacks: dict[str, dict[str, list[str]]] = {}
            self._project_configs: dict[str, object] = {}

        async def _open_registration_store(self) -> RegistrationRedisStore:
            return RegistrationRedisStore(self._backend)

        def _deregister_project(self, project_id: str) -> list[str]:
            self._project_tools.pop(project_id, None)
            self._project_graphs.pop(project_id, None)
            self._project_configs.pop(project_id, None)
            return []

        def _register_graph(self, project_id: str, graph_config, **kwargs) -> str:
            _ = kwargs
            name = f"project:{project_id}:{graph_config.name}"
            self._project_graphs.setdefault(project_id, {})[graph_config.name] = name
            return name

    harness = _Harness(redis_backend)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "contextunity.router.modules.tools.register_tool",
            lambda *args, **kwargs: None,
        )
        mp.setattr(
            "contextunity.router.service.mixins.persistence.create_tools_from_bundle",
            lambda _payload, project_id: [],
        )
        restored = await harness._restore_project_from_persistence(_PROJECT_ID)

    assert restored is True
    assert _PROJECT_ID in harness._project_configs
    config = harness._project_configs[_PROJECT_ID]
    assert config.get("allowed_tenants") == list(_TENANTS)
    assert "analytics" in harness._project_graphs[_PROJECT_ID]
    assert "default" in harness._project_graphs[_PROJECT_ID]


@pytest.mark.asyncio
async def test_restore_registrations_skips_hash_and_stream_keys(
    redis_backend: _InMemoryRedis,
) -> None:
    """Startup scan must not treat ``:hash`` / ``:stream`` keys as bundle payloads."""
    store = RegistrationRedisStore(redis_backend)
    payload = _sample_registration_payload()
    await store.persist(_PROJECT_ID, payload)
    await store.write_hash(_PROJECT_ID, _MANIFEST_HASH)
    await store.write_stream_secret(_PROJECT_ID, _STREAM_SECRET)

    restore_calls: list[str] = []

    class _ScanHarness(PersistenceMixin):
        def __init__(self, backend: _InMemoryRedis) -> None:
            self._backend = backend
            self._project_tools: dict[str, list[str]] = {}
            self._project_graphs: dict[str, dict[str, str]] = {}
            self._project_router_callbacks: dict[str, dict[str, list[str]]] = {}
            self._project_configs: dict[str, object] = {}

        async def _open_registration_store(self) -> RegistrationRedisStore:
            return RegistrationRedisStore(self._backend)

        def _deregister_project(self, project_id: str) -> list[str]:
            return []

        def _register_graph(self, project_id: str, graph_config, **kwargs) -> str:
            _ = project_id, kwargs
            restore_calls.append(graph_config.name)
            return f"compiled:{graph_config.name}"

    harness = _ScanHarness(redis_backend)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "contextunity.router.modules.tools.register_tool",
            lambda *args, **kwargs: None,
        )
        mp.setattr(
            "contextunity.router.service.mixins.persistence.create_tools_from_bundle",
            lambda _payload, project_id: [],
        )
        await harness.restore_registrations()

    assert restore_calls == ["analytics"]
    assert _PROJECT_ID in harness._project_configs

    # Wire bytes must round-trip through core json_dumps (no datetime leakage).
    wire = redis_backend.data[store.project_key(_PROJECT_ID)]
    assert wire == json_dumps(payload, default=str)
