from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from contextunity.router.cortex.compiler.platform_tools.intent import (
    detect_intent,
)


@pytest.mark.asyncio
async def test_intent_analyzer_with_data_sources():
    """Given data_sources in manifest config, it includes them in prompt and selects them."""
    # Setup mock LLM
    mock_llm = AsyncMock()
    # Return JSON with selected_sources
    mock_resp = type(
        "MockResp", (), {"text": '{"intent": "rag", "selected_sources": ["wiki_vector"]}'}
    )
    mock_llm.generate.return_value = mock_resp

    state = {
        "messages": [HumanMessage(content="What is the company policy?")],
        "config": {
            "data_sources": [
                {"binding": "wiki_vector", "type": "vector", "description": "Corporate wiki"},
                {"binding": "sales_sql", "type": "sql", "description": "Sales database"},
            ]
        },
        "__manifest_node_config__": {"model": "mock-model"},
    }

    with patch(
        "contextunity.router.modules.models.model_registry.create_llm", return_value=mock_llm
    ):
        result = await detect_intent(state)

    # Must invoke LLM
    mock_llm.generate.assert_called_once()
    prompt_used = mock_llm.generate.call_args[0][0].parts[0].text

    # Prompt must contain the data sources
    assert "wiki_vector" in prompt_used
    assert "Corporate wiki" in prompt_used
    assert "sales_sql" in prompt_used

    # Result must include selected_sources
    assert "dynamic" in result
    dyn = result["dynamic"]
    assert "selected_sources" in dyn
    assert dyn["selected_sources"] == ["wiki_vector"]


@pytest.mark.asyncio
async def test_intent_analyzer_no_sources():
    """If no data_sources provided, returns empty selected_sources and keeps existing flow."""
    mock_llm = AsyncMock()
    mock_resp = type("MockResp", (), {"text": '{"intent": "rag"}'})
    mock_llm.generate.return_value = mock_resp

    state = {
        "messages": [HumanMessage(content="Hello world")],
        "config": {},  # No data sources
    }

    with patch(
        "contextunity.router.modules.models.model_registry.create_llm", return_value=mock_llm
    ):
        result = await detect_intent(state)

    assert "dynamic" in result
    dyn = result["dynamic"]
    assert "selected_sources" in dyn
    assert dyn["selected_sources"] == []
