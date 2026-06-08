"""BrainClient Protocol and factory -- typed interface for RAG pipeline / Brain gRPC calls."""

from typing import Protocol, runtime_checkable

from contextunity.core import ContextToken
from contextunity.core.sdk.models import SearchResult
from contextunity.core.sdk.responses import EpisodeRecord
from contextunity.core.types import ContextUnitPayload, JsonDict


class BrainClientProtocol(Protocol):
    """Structural type matching BrainClient SDK surface used by platform tools."""

    # KnowledgeMixin
    async def search(
        self,
        tenant_id: str,
        query_text: str,
        limit: int = 5,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Perform semantic vector search over tenant knowledge base.

        Args:
            tenant_id: Tenant scope for multi-tenant isolation.
            query_text: Natural-language query for embedding similarity.
            limit: Maximum number of results to return.
            source_types: Optional filter by document source type.

        Returns:
            Ranked list of search results with content and metadata.
        """
        ...

    async def upsert(
        self,
        tenant_id: str,
        content: str,
        source_type: str,
        metadata: JsonDict | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Insert or update a document in the tenant knowledge base.

        Args:
            tenant_id: Tenant scope for isolation.
            content: Document text to index.
            source_type: Classification tag (e.g., ``"faq"``, ``"article"``).
            metadata: Optional key-value metadata attached to the document.
            doc_id: Optional stable document ID for idempotent updates.

        Returns:
            Persisted document ID.
        """
        ...

    # MemoryMixin
    async def add_episode(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        content: str,
        session_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> str:
        """Persist a new episodic memory record.

        Returns:
            Episode ID string.
        """
        ...

    async def get_recent_episodes(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 5,
    ) -> list[EpisodeRecord]:
        """Retrieve recent episodic memories for a user.

        Returns:
            Chronologically ordered list of episode dicts.
        """
        ...

    async def get_user_facts(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> JsonDict:
        """Retrieve entity facts about a user from the knowledge graph.

        Returns:
            Dict of fact key-value pairs.
        """
        ...


@runtime_checkable
class BrainBlackboardWriter(Protocol):
    """Optional Brain client capability for blackboard writes."""

    async def write_blackboard(
        self,
        *,
        tenant_id: str,
        scope_path: str,
        content: ContextUnitPayload,
        ttl_seconds: int | None = None,
        created_by: str | None = None,
    ) -> JsonDict:
        """Write content into the blackboard."""
        ...


@runtime_checkable
class BrainBlackboardReader(Protocol):
    """Optional Brain client capability for blackboard reads."""

    async def read_blackboard(self, *, ids: list[str], tenant_id: str) -> JsonDict:
        """Read blackboard items by ID."""
        ...


@runtime_checkable
class BrainKnowledgeGraphClient(Protocol):
    """Optional Brain client capability for knowledge-graph queries."""

    async def query_kg(
        self,
        *,
        tenant_id: str,
        entity: str,
        direction: str,
        depth: int,
    ) -> object:
        """Query the knowledge graph."""
        ...


def get_brain_client(tenant_id: str, access_token: ContextToken) -> BrainClientProtocol:
    """Construct a BrainClient scoped to a specific tenant and caller token."""
    from contextunity.core.sdk import BrainClient

    return BrainClient(
        tenant_id=tenant_id,
        token=access_token,
    )


_get_brain_client = get_brain_client
