"""Synthesizer platform tool for LangGraph."""

import logging
import time
from typing import ClassVar

from contextunity.core.types import is_object_list
from pydantic import BaseModel, ConfigDict

from contextunity.router.cortex.types import GraphState, StateUpdate
from contextunity.router.cortex.utils.pipeline import pipeline_log
from contextunity.router.modules.models.types import ModelRequest, TextPart
from contextunity.router.modules.observability import retrieval_span

logger = logging.getLogger(__name__)


class SynthesizerConfig(BaseModel):
    """Configuration for the synthesizer platform tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    output_mode: str = "direct"
    model: str | None = None
    prompt_ref: str | None = None


async def synthesize_results(state: GraphState) -> StateUpdate:
    """Merge results from multiple data sources into a unified response."""
    from contextunity.router.cortex.config_resolution import get_node_manifest_config
    from contextunity.router.modules.models import model_registry

    # In a Send()-based fanout map-reduce pattern, LangGraph collects the intermediate_results
    # or other mapped values as a list. We need to find the array of partial generation outputs.
    # We will look for them in intermediate_results['fanout_outputs'] or some expected key.

    # For now, a basic pass-through mock implementation.
    intermediate_map = state.get("intermediate_results", {})
    raw_results = intermediate_map.get("fanout_outputs", [])
    results = [str(item) for item in raw_results] if is_object_list(raw_results) else []

    if not results:
        return {"intermediate_results": {"synthesizer": "No results to synthesize."}}

    node_config = get_node_manifest_config(state, "synthesize_results")
    model_key = node_config.get("model") or "vertex/gemini-2.5-flash-lite"
    model = model_registry.create_llm(model_key)

    with retrieval_span(name="synthesizer", input_data={"count": len(results)}):
        t0 = time.perf_counter()

        system_prompt = "You are an expert synthesizer. Combine the following data into a single coherent response."
        user_prompt = "\n\n".join(
            f"SOURCE RESULT {index + 1}:\n{result}" for index, result in enumerate(results)
        )

        # Simple string prompt for our TDD
        request = ModelRequest(system=system_prompt, parts=[TextPart(text=user_prompt)])
        response = await model.generate(request)
        text = response.text

        pipeline_log("synthesizer.out", output_length=len(text), duration=time.perf_counter() - t0)

        return {"intermediate_results": {"synthesized_output": text}}
