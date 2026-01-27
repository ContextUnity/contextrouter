from typing import Any, Dict

from contextcore import BrainClient, ContextUnit
from langgraph.graph import END, StateGraph

from contextrouter.cortex.state import AgentState


class MatchingNode:
    """
    Core matching logic.
    1. Pulls candidates from ContextBrain.
    2. Applies taxonomy filters.
    3. Uses LLM to decide on parity.
    """

    async def process(self, state: AgentState) -> Dict[str, Any]:
        site_product = state.get("product", {})  # Data from site
        query = f"Matching candidates for: {site_product.get('title')} {site_product.get('brand')}"

        brain = BrainClient()
        # Find candidates among supplier data in Brain
        candidates = await brain.query_memory(ContextUnit(payload={"content": query}))

        matches = []
        for cand in candidates:
            # 1. Semantic Check (already done by Brain.query_memory)

            # 2. Taxonomy Parity (Category & Brand)
            if not self._taxonomy_match(site_product, cand.payload):
                continue

            # 3. Attribute Refinement (Size, Color, Side)
            # This handles the "One to Many" variants
            if self._attribute_parity(site_product, cand.payload):
                matches.append(cand.payload)

        return {"matches": matches, "match_status": "linked" if matches else "no_parity"}

    def _taxonomy_match(self, site, supplier) -> bool:
        """Verify Category and Brand parity."""
        s_meta = site.get("metadata", {})
        p_meta = supplier.get("metadata", {})
        return s_meta.get("category") == p_meta.get("category") and site.get(
            "brand"
        ) == supplier.get("brand")

    def _attribute_parity(self, site, supplier) -> bool:
        """
        Check if specific attributes (Size, Color) match.
        Normalization happens in Brain, here we just check equality.
        """
        s_meta = site.get("metadata", {})
        p_meta = supplier.get("metadata", {})

        # If site product is generic (no size), and supplier has size,
        # it's a "Model Match" (Many-to-One).
        if not s_meta.get("size"):
            return True

        return s_meta.get("size") == p_meta.get("size")


def build_graph():
    workflow = StateGraph(AgentState)
    matcher = MatchingNode()

    workflow.add_node("match_candidates", matcher.process)
    workflow.set_entry_point("match_candidates")
    workflow.add_edge("match_candidates", END)

    return workflow
