import json

import pytest

from contextunity.router.service.mixins.persistence import PersistenceMixin
from contextunity.router.service.registration_redis import RegistrationRedisStore


class _FakeRedisBackend:
    def __init__(self, data: dict[str, str]) -> None:
        self.data = data
        self.deleted_calls: list[tuple[str, ...]] = []

    async def set(self, key: str, value: str) -> None:
        self.data[key] = value

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def delete(self, *keys: str) -> int:
        self.deleted_calls.append(keys)
        deleted = 0
        for key in keys:
            if key in self.data:
                deleted += 1
                del self.data[key]
        return deleted

    async def scan_iter(self, match: str):
        for key in list(self.data.keys()):
            if key.startswith("router:registrations:"):
                yield key

    async def aclose(self) -> None:
        return None


class _PersistenceHarness(PersistenceMixin):
    def __init__(self, backend: _FakeRedisBackend) -> None:
        self._backend = backend
        self._project_tools = {}
        self._project_graphs = {}
        self._project_router_callbacks = {}
        self._project_configs = {}
        self.register_calls: list[tuple[str, str]] = []

    async def _open_registration_store(self):
        return RegistrationRedisStore(self._backend)

    def _deregister_project(self, project_id: str):
        return []

    def _register_graph(self, project_id: str, graph_config, **kwargs):
        self.register_calls.append((project_id, graph_config.name))
        return f"project:{project_id}:{graph_config.name}"


@pytest.mark.asyncio
async def test_restore_skips_hash_keys():
    payload = {
        "project_id": "nszu",
        "default_graph": "main",
        "graph": {"main": {"template": "yaml:retrieval_augmented"}},
        "policy": {},
    }
    backend = _FakeRedisBackend(
        {
            "router:registrations:nszu": json.dumps(payload),
            "router:registrations:nszu:hash": "abc123",
            "router:registrations:nszu:stream": "stream-secret",
        }
    )
    mixin = _PersistenceHarness(backend)

    await mixin.restore_registrations()

    assert mixin.register_calls == [("nszu", "main")]
    assert backend.deleted_calls == []


@pytest.mark.asyncio
async def test_restore_deletes_stale_record_with_hash_and_logs_code(caplog):
    payload = {
        "project_id": "nszu",
        "default_graph": "main",
        # Old/invalid shape: GraphEntry expects dict, not raw string.
        "graph": {"main": ""},
        "policy": {},
    }
    backend = _FakeRedisBackend(
        {
            "router:registrations:nszu": json.dumps(payload),
            "router:registrations:nszu:hash": "stalehash",
        }
    )
    mixin = _PersistenceHarness(backend)

    with caplog.at_level("WARNING"):
        await mixin.restore_registrations()

    assert backend.deleted_calls == [
        (
            "router:registrations:nszu",
            "router:registrations:nszu:hash",
            "router:registrations:nszu:stream",
        )
    ]
    assert "CONFIGURATION_ERROR" in caplog.text
    assert "Invalid persisted registration payload for project 'nszu'" in caplog.text


@pytest.mark.asyncio
async def test_restore_registers_tools_from_persisted_bundle():
    payload = {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
        "default_graph": "main",
        "graph": {"main": {"template": "yaml:retrieval_augmented"}},
        "policy": {},
        "tools": [
            {
                "name": "execute_nszu_sql",
                "type": "sql",
                "description": "Execute SQL",
                "config": {"read_only": True, "execution": "federated"},
            }
        ],
    }
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)

    registered: list[tuple[str, str]] = []

    def _track_register(tool_instance, *, allowed_tenants=(), tenant="", project_id=""):
        assert project_id == "nszu"
        registered.append((tool_instance.name, tuple(allowed_tenants) or (tenant,)))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "contextunity.router.modules.tools.register_tool",
            _track_register,
        )
        await mixin.restore_registrations()

    assert ("execute_nszu_sql", ("nszu",)) in registered
    assert "execute_nszu_sql" in mixin._project_tools["nszu"]


@pytest.mark.asyncio
async def test_restore_project_returns_false_when_store_unavailable():
    mixin = _PersistenceHarness(_FakeRedisBackend({}))

    async def _no_store():
        return None

    mixin._open_registration_store = _no_store  # type: ignore[method-assign]
    result = await mixin._restore_project_from_persistence("nszu")
    assert result is False


@pytest.mark.asyncio
async def test_manifest_hash_requires_existing_payload():
    backend = _FakeRedisBackend({"router:registrations:nszu:hash": "orphan"})
    mixin = _PersistenceHarness(backend)

    assert await mixin._check_manifest_hash("nszu", "orphan") is False


@pytest.mark.asyncio
async def test_manifest_hash_reads_atomic_embedded_hash():
    payload = {"project_id": "nszu", "__registration_hash": "current"}
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)

    assert await mixin._check_manifest_hash("nszu", "current") is True


@pytest.mark.asyncio
async def test_restore_project_returns_false_when_no_payload():
    backend = _FakeRedisBackend({})
    mixin = _PersistenceHarness(backend)
    result = await mixin._restore_project_from_persistence("nszu")
    assert result is False


@pytest.mark.asyncio
async def test_restore_project_deregisters_existing_before_restore():
    payload = {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
        "default_graph": "main",
        "graph": {"main": {"template": "yaml:retrieval_augmented"}},
        "policy": {},
    }
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)
    mixin._project_tools["nszu"] = ["old_tool"]

    deregister_calls: list[str] = []
    original = mixin._deregister_project

    def _track(project_id: str):
        deregister_calls.append(project_id)
        return original(project_id)

    mixin._deregister_project = _track  # type: ignore[method-assign]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("contextunity.router.modules.tools.register_tool", lambda *a, **kw: None)
        result = await mixin._restore_project_from_persistence("nszu")

    assert result is True
    assert "nszu" in deregister_calls


@pytest.mark.asyncio
async def test_restore_project_returns_false_when_graph_map_empty():
    from contextunity.router.modules.tools import deregister_tool, list_project_tools

    payload = {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
        "default_graph": "main",
        "graph": {},
        "policy": {},
        "tools": [
            {
                "name": "restore_probe_tool",
                "type": "sql",
                "description": "Probe",
                "config": {"read_only": True, "execution": "federated"},
            }
        ],
    }
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)
    try:
        result = await mixin._restore_project_from_persistence("nszu")
        assert result is False
        assert "restore_probe_tool" not in list_project_tools("nszu")
    finally:
        _ = deregister_tool("restore_probe_tool", project_id="nszu")


@pytest.mark.asyncio
async def test_restore_project_returns_true_and_populates_state():
    payload = {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
        "default_graph": "main",
        "graph": {"main": {"template": "yaml:retrieval_augmented"}},
        "policy": {},
    }
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("contextunity.router.modules.tools.register_tool", lambda *a, **kw: None)
        result = await mixin._restore_project_from_persistence("nszu")

    assert result is True
    assert "nszu" in mixin._project_graphs
    assert "default" in mixin._project_graphs["nszu"]
    assert "nszu" in mixin._project_configs


@pytest.mark.asyncio
async def test_restore_project_returns_false_on_invalid_graph_entry(caplog):
    payload = {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
        "default_graph": "main",
        "graph": {"main": "invalid-not-a-dict"},
        "policy": {},
    }
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)

    with caplog.at_level("WARNING"):
        result = await mixin._restore_project_from_persistence("nszu")

    assert result is False
    assert "Failed to restore project" in caplog.text


@pytest.mark.asyncio
async def test_restore_rejects_multi_graph_without_default_graph(caplog):
    payload = {
        "project_id": "nszu",
        "allowed_tenants": ["nszu"],
        "graph": {
            "main": {"template": "yaml:retrieval_augmented"},
            "secondary": {"template": "yaml:retrieval_augmented"},
        },
        "policy": {},
    }
    backend = _FakeRedisBackend({"router:registrations:nszu": json.dumps(payload)})
    mixin = _PersistenceHarness(backend)

    with caplog.at_level("WARNING"):
        await mixin.restore_registrations()

    assert backend.deleted_calls == [
        (
            "router:registrations:nszu",
            "router:registrations:nszu:hash",
            "router:registrations:nszu:stream",
        )
    ]
    assert "Persisted multi-graph registration missing valid default_graph" in caplog.text
