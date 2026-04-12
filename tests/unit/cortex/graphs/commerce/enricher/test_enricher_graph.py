import pytest

from contextunity.router.cortex.graphs.commerce.enricher import build_enricher_graph


@pytest.fixture
def graph():
    config = {
        "model_key": "openai/gpt-4o-mini",
        "perplexity_model": "perplexity/sonar",
        "reasoning_effort": "low",
    }
    return build_enricher_graph(config)


@pytest.mark.asyncio
async def test_enricher_graph_compilation(graph):
    assert graph is not None
    # Basic validation that it compiled and has target nodes
    nodes = [n for n in graph.nodes]
    assert "init_credentials" in nodes
    assert "normalize_raw" in nodes
    assert "search_images" in nodes
    assert "generate_description" in nodes
    assert "ner_technologies" in nodes
    assert "verify_technologies_bidi" in nodes
    assert "create_missing_technology_articles" in nodes
    assert "map_attributes" in nodes


@pytest.mark.asyncio
async def test_enricher_graph_execution_mocked(graph):
    # Mock the stream manager to avoid real backend calls
    from unittest.mock import AsyncMock, patch

    mock_manager = AsyncMock()
    mock_manager.execute.side_effect = lambda tenant, tool, payload, **kwargs: {
        "verify_technologies": {"missing": ["GORE-TEX"]},
        "create_wagtail_technology": {"id": 999},
        "save_enriched_product": {"success": True},
    }.get(tool, {})

    # Mock ModelRegistry to return deterministic outputs, preventing real API calls
    mock_llm = AsyncMock()

    async def mock_generate(req):
        from contextunity.router.modules.models.types import ModelResponse, ProviderInfo

        sys_text = req.system.lower()

        if "seo metadata" in sys_text:
            text = '{"meta_title": "test", "meta_description": "test", "slug": "test"}'
        elif "description" in sys_text:
            text = "<p>Test description</p>"
        elif "encyclopedia entries" in sys_text:
            text = '{"content": "a waterproof fabric"}'
        elif "product attributes" in sys_text:
            text = '[{"name": "Material", "value": "Nylon"}]'
        elif "taxonomy" in sys_text or "products" in sys_text or "dealer" in sys_text:
            text = '[{"brand": "Arc\'Teryx", "name": "GORE-TEX Alpha Jacket", "id": 123}]'
        else:
            # Fallback for NER technologies
            text = '["GORE-TEX"]'

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(provider="mock", model_name="mock", model_key="mock"),
        )

    mock_llm.generate.side_effect = mock_generate

    with (
        patch(
            "contextunity.router.service.stream_executors.get_stream_executor_manager",
            return_value=mock_manager,
        ),
        patch(
            "contextunity.router.modules.models.registry.model_registry.create_llm",
            return_value=mock_llm,
        ),
    ):
        initial_state = {
            "tenant_id": "test_tenant",
            "dealer_products": [{"id": 123, "name": "GORE-TEX Alpha Jacket", "brand": "Arc'Teryx"}],
            "extracted_technologies_names": ["GORE-TEX"],
        }

        result = await graph.ainvoke(initial_state)

        # Assert successful flow completion
        assert "trace_id" in result
        assert result["step_traces"]

        # Verify outputs from the pipeline steps
        assert result["google_search_urls"]
        assert 123 in result["google_search_urls"]

        assert result["descriptions"]
        assert 123 in result["descriptions"]
        assert result["descriptions"][123]["uk"]

        assert "GORE-TEX" in result["extracted_technologies_names"]

        # Check BiDi mocked impact
        assert "GORE-TEX" in result["missing_technologies"]
        assert 999 in result["created_technologies_ids"]
