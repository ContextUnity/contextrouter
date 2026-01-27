import logging

from langchain_core.messages import AIMessage, HumanMessage

from contextrouter.core import get_core_config
from contextrouter.core.memory import MemoryManager
from contextrouter.cortex import AgentState

logger = logging.getLogger(__name__)


async def reflect_interaction(state: AgentState) -> dict:
    """
    Analyzes the finished interaction to update Episodic and Entity memory.
    Runs at the end of the graph flow.
    """
    manager = MemoryManager(get_core_config())
    user_id = state.metadata.get("user_id", "anonymous")
    session_id = state.session_id

    # 1. Record Episode (What happened)
    last_user_msg = ""
    last_ai_msg = ""

    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and not last_user_msg:
            last_user_msg = msg.content
        if isinstance(msg, AIMessage) and not last_ai_msg:
            last_ai_msg = msg.content
        if last_user_msg and last_ai_msg:
            break

    if last_user_msg:
        summary = f"User asked: {last_user_msg[:100]}... AI responded: {last_ai_msg[:100]}..."
        await manager.record_episode(
            user_id=user_id,
            content=summary,
            session_id=session_id,
            metadata={"full_query": last_user_msg},
        )

    # 2. Extract Facts (Simplified demonstration)
    # In a real scenario, we'd call an LLM here to 'distill' facts.
    # Demonstration: if user mentions a color preference
    if "люблю" in last_user_msg.lower() or "подобається" in last_user_msg.lower():
        # Heuristic for color/brand extraction from new metadata
        for color in ["синій", "червоний", "чорний"]:
            if color in last_user_msg.lower():
                await manager.upsert_user_fact(
                    user_id=user_id, key="preferred_color", value=color, confidence=0.8
                )
                logger.info(f"Memory: Extracted preference for {color}")

    return {}
