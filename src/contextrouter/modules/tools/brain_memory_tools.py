"""Brain Memory Tools for Dispatcher Agent.

Provides persistent memory operations backed by ContextBrain via SDK.
These tools expose episodic memory and entity (fact) storage to LLM agents.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Singleton BrainClient instance
_brain_client = None


def _get_brain_client():
    """Get or create BrainClient singleton."""
    global _brain_client
    if _brain_client is None:
        from contextcore.permissions import Permissions
        from contextcore.sdk import BrainClient
        from contextcore.tokens import ContextToken

        from contextrouter.core import get_core_config

        brain_host = get_core_config().brain.grpc_endpoint
        token = ContextToken(
            token_id="router-memory-service",
            permissions=(
                Permissions.MEMORY_WRITE,
                Permissions.MEMORY_READ,
            ),
        )
        _brain_client = BrainClient(host=brain_host, mode="grpc", token=token)
    return _brain_client


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
) -> dict[str, Any]:
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
        brain = _get_brain_client()
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
) -> dict[str, Any]:
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
        brain = _get_brain_client()
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
) -> dict[str, Any]:
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
        brain = _get_brain_client()
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
) -> dict[str, Any]:
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
        brain = _get_brain_client()
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
