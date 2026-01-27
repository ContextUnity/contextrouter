from typing import Any, Dict

from contextcore import BrainClient, ContextUnit

from contextrouter.core.interfaces import BaseAgent


class RetrievalNode(BaseAgent):
    """
    LangGraph node that delegates retrieval to ContextBrain.
    """

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch context from Brain and update state.
        """
        query = state.get("query", "")
        # Get Brain URL from config/env
        brain_client = BrainClient(host="localhost:50051")

        unit = ContextUnit(payload={"content": query}, modality="text")

        # Call Brain
        results = await brain_client.query_memory(unit)

        # Update state with retrieved context
        return {
            "context": [res.payload for res in results],
            "messages": state.get("messages", [])
            + [{"role": "system", "content": "Context fetched from Brain memory."}],
        }
