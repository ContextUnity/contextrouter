from contextunity.core import get_contextunit_logger

from contextunity.router.core import get_core_config
from contextunity.router.core.memory import MemoryManager
from contextunity.router.cortex import AgentState

logger = get_contextunit_logger(__name__)


async def fetch_memory(state: AgentState) -> dict:
    """
    Explicitly fetch memory context before intent detection.
    """
    manager = MemoryManager(get_core_config())

    # In a real scenario, use user_id from token/context
    user_id = state.metadata.get("user_id", "anonymous")

    context = await manager.compile_context(
        user_id=user_id, query=state.user_query, session_id=state.session_id
    )

    return {"memory_context": context}
