"""Memory-fetch platform tool — pre-LLM episodic/fact context injection."""

from contextunity.core import get_contextunit_logger

from contextunity.router.core import get_core_config
from contextunity.router.core.memory import MemoryManager
from contextunity.router.cortex.types import GraphState, StateUpdate

logger = get_contextunit_logger(__name__)


async def fetch_memory(state: GraphState) -> StateUpdate:
    """Explicitly fetch memory context before intent detection."""
    manager = MemoryManager(get_core_config())

    # In a real scenario, use user_id from token/context
    metadata = state.get("metadata") or {}
    user_id = "anonymous"
    candidate = metadata.get("user_id")
    if isinstance(candidate, str) and candidate.strip():
        user_id = candidate

    dyn = state.get("dynamic", {})
    context = await manager.compile_context(
        user_id=user_id,
        query=str(dyn.get("user_query") or ""),
        session_id=str(state.get("session_id") or ""),
    )

    return {"dynamic": {"memory_context": context}}
