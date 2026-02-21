"""Memory manager — orchestrates Semantic, Episodic, and Entity memory layers.

Uses BrainClient SDK for all persistent memory operations.
"""

import logging
from typing import Any, Dict, Optional

from contextcore.sdk import BrainClient

from .config import Config

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates Semantic, Episodic, and Entity memory layers.

    Builds context blocks for LLM nodes using real Brain data.
    """

    def __init__(self, config: Config):
        self.config = config
        brain_host = getattr(config.providers, "brain_host", "localhost:50051")

        from contextrouter.core.brain_token import get_brain_service_token

        self.brain = BrainClient(host=brain_host, token=get_brain_service_token())

    async def compile_context(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        tenant_id: str = "default",
    ) -> str:
        """Assemble a system context block from multiple memory layers.

        Args:
            user_id: User identifier.
            query: Current user query (for semantic search).
            session_id: Optional session identifier.
            tenant_id: Tenant scope.

        Returns:
            Formatted context string for LLM system prompt.
        """
        context_parts = []

        # 1. Semantic Memory (RAG Chunks)
        try:
            semantic_results = await self.brain.search(
                tenant_id=tenant_id,
                query_text=query,
                limit=5,
            )
            if semantic_results:
                chunks = "\n".join([r.content for r in semantic_results if r.content])
                if chunks:
                    context_parts.append(f"### Relevant Knowledge\n{chunks}")
        except Exception as e:
            logger.error("Failed to fetch semantic memory: %s", e)

        # 2. Episodic Memory (Recent conversation episodes)
        try:
            episodes = await self.brain.get_recent_episodes(
                tenant_id=tenant_id,
                user_id=user_id,
                limit=5,
            )
            if episodes:
                episode_lines = []
                for ep in episodes:
                    ts = ep.get("created_at", "")
                    content = ep.get("content", "")
                    if content:
                        episode_lines.append(f"- [{ts}] {content}")
                if episode_lines:
                    context_parts.append("### Recent History\n" + "\n".join(episode_lines))
        except Exception as e:
            logger.error("Failed to fetch episodic memory: %s", e)

        # 3. Entity Memory (User Facts)
        try:
            facts = await self.brain.get_user_facts(
                tenant_id=tenant_id,
                user_id=user_id,
            )
            if facts:
                fact_lines = [f"- {k}: {v}" for k, v in facts.items()]
                context_parts.append("### User Facts\n" + "\n".join(fact_lines))
        except Exception as e:
            logger.error("Failed to fetch entity memory: %s", e)

        return "\n\n".join(context_parts) if context_parts else ""

    async def record_episode(
        self,
        user_id: str,
        content: str,
        session_id: str,
        metadata: Optional[Dict] = None,
        tenant_id: str = "default",
    ) -> str:
        """Persist a new interaction to episodic memory via Brain.

        Args:
            user_id: User identifier.
            content: Episode content.
            session_id: Session identifier.
            metadata: Additional metadata.
            tenant_id: Tenant scope.

        Returns:
            Episode ID.
        """
        try:
            episode_id = await self.brain.add_episode(
                tenant_id=tenant_id,
                user_id=user_id,
                content=content,
                session_id=session_id,
                metadata=metadata,
            )
            logger.info("Recorded episode %s for user %s", episode_id, user_id)
            return episode_id
        except Exception as e:
            logger.error("Failed to record episode: %s", e)
            return ""

    async def upsert_user_fact(
        self,
        user_id: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        tenant_id: str = "default",
    ) -> None:
        """Store a persistent fact about the user.

        Args:
            user_id: User identifier.
            key: Fact key (e.g., "language", "specialty").
            value: Fact value.
            confidence: Confidence score (0-1).
            tenant_id: Tenant scope.
        """
        try:
            await self.brain.upsert_fact(
                tenant_id=tenant_id,
                user_id=user_id,
                key=key,
                value=value,
                confidence=confidence,
            )
            logger.info("Upserted fact %s=%s for user %s", key, value, user_id)
        except Exception as e:
            logger.error("Failed to upsert fact: %s", e)
