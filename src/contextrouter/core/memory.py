import logging
from typing import Any, Dict, Optional

from contextcore.sdk import BrainClient, ContextUnit

from .config import Config

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Orchestrates Semantic, Episodic, and Entity memory layers.
    Builds context blocks for LLM nodes.
    """

    def __init__(self, config: Config):
        self.config = config
        # Assuming Brain host is configurable
        brain_host = getattr(config.providers, "brain_host", "localhost:50051")
        self.brain = BrainClient(host=brain_host)

    async def compile_context(
        self, user_id: str, query: str, session_id: Optional[str] = None
    ) -> str:
        """
        Assembles a system context block from multiple memory layers.
        """
        context_parts = []

        # 1. Semantic Memory (RAG Chunks)
        # Using a generic unit to query Brain
        query_unit = ContextUnit(
            payload={"query": query}, security={"read": ["knowledge:read"]}, modality="text"
        )
        try:
            semantic_results = await self.brain.query_memory(query_unit)
            if semantic_results:
                context_parts.append(
                    "### Relevant Knowledge\n"
                    + "\n".join([r.payload.get("content", "") for r in semantic_results])
                )
        except Exception as e:
            logger.error(f"Failed to fetch semantic memory: {e}")

        # 2. Episodic Memory (Recent chat history from Brain's Journal)
        # Assuming we have a gRPC method for history in the future, or use query_memory with filters
        # For now, placeholder for recent episodes
        context_parts.append("### Recent History\n(Session history placeholder)")

        # 3. Entity Memory (User Facts)
        # context_parts.append("### User Facts\n- Prefer metric units\n- Language: Ukrainian")

        return "\n\n".join(context_parts)

    async def record_episode(
        self, user_id: str, content: str, session_id: str, metadata: Optional[Dict] = None
    ):
        """Persist a new interaction to episodic memory via Brain."""
        unit = ContextUnit(
            payload={
                "content": content,
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata or {},
            },
            modality="text",
        )
        # Assuming sdk.BrainClient is updated or we use raw stub
        if hasattr(self.brain, "add_episode"):
            await self.brain.add_episode(unit)
        else:
            logger.warning(
                "BrainClient does not support add_episode yet. Falling back to memorize."
            )
            await self.brain.memorize(unit)

    async def upsert_user_fact(self, user_id: str, key: str, value: Any, confidence: float = 1.0):
        """Store a persistent fact about the user."""
        unit = ContextUnit(
            payload={"user_id": user_id, "key": key, "value": value, "confidence": confidence},
            modality="text",
        )
        if hasattr(self.brain, "upsert_fact"):
            await self.brain.upsert_fact(unit)
