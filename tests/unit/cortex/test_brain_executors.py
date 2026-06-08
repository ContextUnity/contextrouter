"""Behavioral tests for Brain platform tool executors.

Uses a FakeBrainClient (satisfies _BrainClientProtocol) to test
executor logic: query extraction, tenant isolation, security guards,
content serialization — without gRPC or mocks.
"""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import SecurityError
from contextunity.core.sdk.models import SearchResult
from contextunity.core.tokens import ContextToken

from contextunity.router.cortex.compiler.platform_tools.brain.executors import (
    _brain_kg_query_executor,
    _brain_memory_read_executor,
    _brain_memory_write_executor,
    _brain_search_executor,
    _brain_upsert_executor,
)
from contextunity.router.cortex.compiler.platform_tools.brain.schemas import (
    BrainKGQueryConfig,
    BrainMemoryReadConfig,
    BrainMemoryWriteConfig,
    BrainSearchConfig,
    BrainUpsertConfig,
)

# ── Fake BrainClient ──────────────────────────────────────────────────────


class FakeBrainClient:
    """In-memory brain client satisfying _BrainClientProtocol."""

    def __init__(self):
        self.search_calls: list[dict] = []
        self.upsert_calls: list[dict] = []
        self.add_episode_calls: list[dict] = []
        self.get_recent_episodes_calls: list[dict] = []
        self.get_user_facts_calls: list[dict] = []
        self._docs: list[SearchResult] = []
        self._episodes: list[dict] = []
        self._facts: dict = {}

    async def search(
        self,
        tenant_id: str,
        query_text: str,
        limit: int = 5,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            {
                "tenant_id": tenant_id,
                "query_text": query_text,
                "limit": limit,
                "source_types": source_types,
            }
        )
        return self._docs[:limit]

    async def upsert(
        self,
        tenant_id: str,
        content: str,
        source_type: str,
        metadata=None,
        doc_id=None,
    ) -> str:
        self.upsert_calls.append(
            {
                "tenant_id": tenant_id,
                "content": content,
                "source_type": source_type,
                "metadata": metadata,
            }
        )
        return "doc-001"

    async def add_episode(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        content: str,
        session_id: str | None = None,
        metadata=None,
    ) -> str:
        self.add_episode_calls.append(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "content": content,
                "session_id": session_id,
            }
        )
        return "ep-001"

    async def get_recent_episodes(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 5,
    ) -> list[dict]:
        self.get_recent_episodes_calls.append(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "limit": limit,
            }
        )
        return self._episodes[:limit]

    async def get_user_facts(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> dict:
        self.get_user_facts_calls.append(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
            }
        )
        return self._facts


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_brain():
    return FakeBrainClient()


@pytest.fixture()
def _patch_brain_client(monkeypatch, fake_brain):
    """Replace _get_brain_client with FakeBrainClient."""
    monkeypatch.setattr(
        "contextunity.router.cortex.compiler.platform_tools.brain.executors._get_brain_client",
        lambda tenant_id, access_token: fake_brain,
    )


def _state_with_token(*, user_id="user-1", tenant_id="acme", **extra):
    """Build a minimal state dict with __token__ (SPOT pattern)."""
    token = ContextToken(
        token_id="t",
        user_id=user_id,
        permissions=("brain:read", "brain:write", "tool:*"),
        allowed_tenants=(tenant_id,),
    )
    return {"__token__": token, "tenant_id": tenant_id, **extra}


# ── Tests: brain_search ──────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_brain_client")
class TestBrainSearchExecutor:
    @pytest.mark.asyncio
    async def test_extracts_query_from_dynamic_bucket(self, fake_brain):
        state = _state_with_token()
        state["dynamic"] = {"query": "find stuff"}
        result = await _brain_search_executor(state, BrainSearchConfig())
        assert fake_brain.search_calls[0]["query_text"] == "find stuff"
        assert result["query"] == "find stuff"

    @pytest.mark.asyncio
    async def test_fallback_to_last_message(self, fake_brain):
        state = _state_with_token(
            messages=[{"role": "user", "content": "hello brain"}],
        )
        await _brain_search_executor(state, BrainSearchConfig())
        assert fake_brain.search_calls[0]["query_text"] == "hello brain"

    @pytest.mark.asyncio
    async def test_empty_state_uses_empty_query(self, fake_brain):
        state = _state_with_token()
        await _brain_search_executor(state, BrainSearchConfig())
        assert fake_brain.search_calls[0]["query_text"] == ""

    @pytest.mark.asyncio
    async def test_top_k_propagated(self, fake_brain):
        state = _state_with_token()
        await _brain_search_executor(state, BrainSearchConfig(top_k=3))
        assert fake_brain.search_calls[0]["limit"] == 3

    @pytest.mark.asyncio
    async def test_collection_filter(self, fake_brain):
        state = _state_with_token()
        await _brain_search_executor(state, BrainSearchConfig(collection="docs"))
        assert fake_brain.search_calls[0]["source_types"] == ["docs"]

    @pytest.mark.asyncio
    async def test_tenant_id_from_state(self, fake_brain):
        state = _state_with_token(tenant_id="proj-x")
        await _brain_search_executor(state, BrainSearchConfig())
        assert fake_brain.search_calls[0]["tenant_id"] == "proj-x"


# ── Tests: brain_memory_read ─────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_brain_client")
class TestBrainMemoryReadExecutor:
    @pytest.mark.asyncio
    async def test_reads_episodes_and_facts(self, fake_brain):
        fake_brain._episodes = [{"id": "ep1", "content": "hello"}]
        fake_brain._facts = {"name": "Alice"}
        state = _state_with_token()
        result = await _brain_memory_read_executor(state, BrainMemoryReadConfig())
        assert result["episodes"] == [{"id": "ep1", "content": "hello"}]
        assert result["facts"] == {"name": "Alice"}
        assert result["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_rejects_token_without_user_id(self, fake_brain):
        state = _state_with_token(user_id=None)
        with pytest.raises(SecurityError, match="user-bound"):
            await _brain_memory_read_executor(state, BrainMemoryReadConfig())

    @pytest.mark.asyncio
    async def test_rejects_cross_user_config_override(self, fake_brain):
        state = _state_with_token(user_id="user-1")
        with pytest.raises(SecurityError, match="must match"):
            await _brain_memory_read_executor(state, BrainMemoryReadConfig(user_id="user-2"))

    @pytest.mark.asyncio
    async def test_limit_propagated(self, fake_brain):
        state = _state_with_token()
        await _brain_memory_read_executor(state, BrainMemoryReadConfig(last_n=3))
        assert fake_brain.get_recent_episodes_calls[0]["limit"] == 3


# ── Tests: brain_memory_write ────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_brain_client")
class TestBrainMemoryWriteExecutor:
    @pytest.mark.asyncio
    async def test_writes_episode_from_final_output(self, fake_brain):
        state = _state_with_token(final_output="important discovery")
        result = await _brain_memory_write_executor(state, BrainMemoryWriteConfig())
        assert result["success"] is True
        assert result["episode_id"] == "ep-001"
        assert fake_brain.add_episode_calls[0]["content"] == "important discovery"

    @pytest.mark.asyncio
    async def test_writes_episode_from_messages_fallback(self, fake_brain):
        state = _state_with_token(
            messages=[{"role": "assistant", "content": "response text"}],
        )
        await _brain_memory_write_executor(state, BrainMemoryWriteConfig())
        assert "response text" in fake_brain.add_episode_calls[0]["content"]

    @pytest.mark.asyncio
    async def test_rejects_token_without_user_id(self, fake_brain):
        state = _state_with_token(user_id=None)
        with pytest.raises(SecurityError, match="user-bound"):
            await _brain_memory_write_executor(state, BrainMemoryWriteConfig())

    @pytest.mark.asyncio
    async def test_rejects_cross_user_config_override(self, fake_brain):
        state = _state_with_token(user_id="user-1")
        with pytest.raises(SecurityError, match="must match"):
            await _brain_memory_write_executor(state, BrainMemoryWriteConfig(user_id="user-2"))

    @pytest.mark.asyncio
    async def test_session_id_propagated(self, fake_brain):
        state = _state_with_token(session_id="sess-42")
        await _brain_memory_write_executor(state, BrainMemoryWriteConfig())
        assert fake_brain.add_episode_calls[0]["session_id"] == "sess-42"


# ── Tests: brain_upsert ─────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_brain_client")
class TestBrainUpsertExecutor:
    @pytest.mark.asyncio
    async def test_upserts_string_content(self, fake_brain):
        state = _state_with_token(final_output="new document text")
        result = await _brain_upsert_executor(state, BrainUpsertConfig())
        assert result["success"] is True
        assert fake_brain.upsert_calls[0]["content"] == "new document text"

    @pytest.mark.asyncio
    async def test_upserts_dict_content_as_string(self, fake_brain):
        state = _state_with_token(final_output={"key": "value"})
        await _brain_upsert_executor(state, BrainUpsertConfig())
        assert "key" in fake_brain.upsert_calls[0]["content"]

    @pytest.mark.asyncio
    async def test_collection_propagated(self, fake_brain):
        state = _state_with_token(final_output="doc")
        await _brain_upsert_executor(state, BrainUpsertConfig(collection="my_collection"))
        assert fake_brain.upsert_calls[0]["source_type"] == "my_collection"


# ── Tests: brain_kg_query (requires getattr guard) ───────────────────────


@pytest.mark.usefixtures("_patch_brain_client")
class TestBrainKGQueryExecutor:
    @pytest.mark.asyncio
    async def test_raises_when_query_kg_not_available(self, fake_brain):
        """query_kg is a planned SDK method — raises PlatformServiceError."""
        from contextunity.core.exceptions import PlatformServiceError

        state = _state_with_token()
        with pytest.raises(PlatformServiceError, match="query_kg"):
            await _brain_kg_query_executor(state, BrainKGQueryConfig())

    @pytest.mark.asyncio
    async def test_calls_query_kg_when_available(self, fake_brain):
        """When query_kg is available, it's called with entity/direction/depth."""

        async def fake_query_kg(*, tenant_id, entity, direction, depth):
            return [{"entity": entity, "relations": []}]

        fake_brain.query_kg = fake_query_kg

        state = _state_with_token()
        state["dynamic"] = {"entity": "Alice"}
        result = await _brain_kg_query_executor(
            state,
            BrainKGQueryConfig(direction="outbound", depth=2),
        )
        assert result["entity"] == "Alice"
        assert result["results"][0]["entity"] == "Alice"
