"""Tests for synthesizer node (Phase 6.4)."""

from unittest.mock import AsyncMock, patch

import pytest

from contextunity.router.cortex.compiler.platform_tools.synthesizer import (
    synthesize_results,
)


@pytest.mark.asyncio
async def test_synthesizer_passes_through_if_no_results():
    """If there are no results to synthesize, it returns a placeholder."""
    state = {"intermediate_results": {}}
    result = await synthesize_results(state)
    assert "synthesizer" in result["intermediate_results"]
    assert result["intermediate_results"]["synthesizer"] == "No results to synthesize."


@pytest.mark.asyncio
async def test_synthesizer_merges_multiple_results():
    """Given results from multiple sources, it merges them using an LLM."""
    state = {
        "intermediate_results": {"fanout_outputs": ["result A", "result B"]},
        "__manifest_node_config__": {"model": "my_model"},
    }

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = type("MockResp", (), {"text": "Merged AB!", "usage": None})()

    with patch(
        "contextunity.router.modules.models.model_registry.create_llm", return_value=mock_llm
    ):
        result = await synthesize_results(state)

    mock_llm.generate.assert_called_once()
    assert "synthesized_output" in result["intermediate_results"]
    assert result["intermediate_results"]["synthesized_output"] == "Merged AB!"
