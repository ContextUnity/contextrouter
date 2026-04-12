"""Brain Memory Tools for Dispatcher Agent.

Provides persistent memory operations backed by cu.brain via SDK.
These tools expose episodic memory and entity (fact) storage to LLM agents.
"""

from __future__ import annotations

from typing import Any

from contextunity.core import get_contextunit_logger
from langchain_core.tools import tool

from contextunity.router.modules.tools.schemas import MemoryRecallResult, MemoryStoreResult

logger = get_contextunit_logger(__name__)

# Per-tenant BrainClient cache — each tenant gets a properly scoped client
_brain_clients: dict[str, object] = {}


def _get_brain_client(tenant_id: str):
    """Get or create BrainClient for a specific tenant.

    Uses the client's **pre-serialized token string** from the current
    auth context. The token is already signed by the project's backend
    (HMAC or Shield) — no Router-level signing backend needed.

    Memory operations require ``memory:write``, which is granted to
    the primary client token.
    """
    if tenant_id not in _brain_clients:
        from contextunity.core.sdk import BrainClient

        _brain_clients[tenant_id] = BrainClient(
            tenant_id=tenant_id,
            token=lambda: _get_auth_token_string(),
        )
    return _brain_clients[tenant_id]


def _get_auth_token_string() -> str | None:
    """Extract the pre-serialized token string from the current gRPC auth context."""
    try:
        from contextunity.core.authz.context import get_auth_context

        ctx = get_auth_context()
        return ctx.token_string if ctx else None
    except Exception:
        return None


# ============================================================================
# Episodic Memory Tools
# ============================================================================


@tool
async def remember_episode(
    content: str,
    user_id: str,
    session_id: str,
    tenant_id: str = "default",
    metadata: dict[str, Any] | None = None,
) -> MemoryStoreResult:
    """Store a conversation episode in long-term memory.

    Use this to persist important interaction moments that should be
    remembered across sessions. Good candidates for episodes:
    - Key decisions made by the user
    - Important questions and their answers
    - Completed tasks or milestones
    - User preferences expressed during conversation

    Args:
        content: Episode summary (what happened)
        user_id: User identifier
        session_id: Current session identifier
        tenant_id: Tenant identifier for isolation
        metadata: Optional additional context

    Returns:
        Dict with episode_id and success status
    """
    try:
        brain = _get_brain_client(tenant_id)
        episode_id = await brain.add_episode(
            tenant_id=tenant_id,
            user_id=user_id,
            content=content,
            session_id=session_id,
            metadata=metadata,
        )
        logger.info("Stored episode %s for user %s", episode_id, user_id)
        return {
            "success": True,
            "episode_id": episode_id,
            "user_id": user_id,
        }
    except Exception as e:
        logger.error("Failed to store episode: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def recall_episodes(
    user_id: str,
    tenant_id: str = "default",
    limit: int = 5,
) -> MemoryStoreResult:
    """Recall recent conversation episodes from long-term memory.

    Use this to retrieve past interactions with a user, providing
    conversational continuity across sessions.

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier for isolation
        limit: Maximum number of episodes to retrieve (default: 5)

    Returns:
        Dict with list of episodes (content, created_at, metadata)
    """
    try:
        brain = _get_brain_client(tenant_id)
        episodes = await brain.get_recent_episodes(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
        )
        logger.info("Retrieved %d episodes for user %s", len(episodes), user_id)
        return {
            "success": True,
            "episodes": episodes,
            "count": len(episodes),
            "user_id": user_id,
        }
    except Exception as e:
        logger.error("Failed to recall episodes: %s", e)
        return {"success": False, "error": str(e), "episodes": []}


# ============================================================================
# Entity Memory (User Facts) Tools
# ============================================================================


@tool
async def learn_user_fact(
    user_id: str,
    key: str,
    value: str,
    tenant_id: str = "default",
    confidence: float = 1.0,
) -> MemoryRecallResult:
    """Store a persistent fact about the user.

    Use this to remember user preferences, attributes, and other
    long-term knowledge. Facts persist indefinitely and are
    automatically updated if the same key is stored again.

    Good candidates for facts:
    - User language preference
    - Professional role / specialty
    - Communication style preference
    - Known constraints or requirements

    Args:
        user_id: User identifier
        key: Fact key (e.g., "language", "specialty", "timezone")
        value: Fact value
        tenant_id: Tenant identifier for isolation
        confidence: Confidence score 0-1 (default: 1.0 = certain)

    Returns:
        Dict with success status
    """
    try:
        brain = _get_brain_client(tenant_id)
        await brain.upsert_fact(
            tenant_id=tenant_id,
            user_id=user_id,
            key=key,
            value=value,
            confidence=confidence,
        )
        logger.info("Stored fact %s=%s for user %s", key, value, user_id)
        return {
            "success": True,
            "key": key,
            "value": value,
            "user_id": user_id,
        }
    except Exception as e:
        logger.error("Failed to store fact: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def recall_user_facts(
    user_id: str,
    tenant_id: str = "default",
) -> MemoryRecallResult:
    """Recall all known facts about a user.

    Use this to retrieve persistent knowledge about a user,
    including preferences, attributes, and context accumulated
    over time from previous interactions.

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier for isolation

    Returns:
        Dict with facts mapping (key -> value)
    """
    try:
        brain = _get_brain_client(tenant_id)
        facts = await brain.get_user_facts(
            tenant_id=tenant_id,
            user_id=user_id,
        )
        logger.info("Retrieved %d facts for user %s", len(facts), user_id)
        return {
            "success": True,
            "facts": facts,
            "count": len(facts),
            "user_id": user_id,
        }
    except Exception as e:
        logger.error("Failed to recall facts: %s", e)
        return {"success": False, "error": str(e), "facts": {}}


__all__ = [
    "remember_episode",
    "recall_episodes",
    "learn_user_fact",
    "recall_user_facts",
]
