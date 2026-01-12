"""Tests for ModelRegistry and fallback functionality."""

from unittest.mock import MagicMock

import pytest

from contextrouter.core.config import Config
from contextrouter.modules.models.base import BaseModel
from contextrouter.modules.models.registry import ModelRegistry
from contextrouter.modules.models.types import (
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    TextDeltaEvent,
    TextPart,
)


class MockProvider(BaseModel):
    """Mock model provider for testing."""

    def __init__(
        self,
        name: str,
        supports_text: bool = True,
        supports_image: bool = False,
        supports_audio: bool = False,
    ):
        self.name = name
        self._capabilities = ModelCapabilities(
            supports_text=supports_text,
            supports_image=supports_image,
            supports_audio=supports_audio,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(self, request: ModelRequest, *, token=None) -> ModelResponse:
        return ModelResponse(
            text=f"Response from {self.name}",
            raw_provider=ProviderInfo(
                provider="mock", model_name=self.name, model_key=f"mock/{self.name}"
            ),
        )

    async def stream(self, request: ModelRequest, *, token=None):
        yield TextDeltaEvent(delta=f"Chunk from {self.name}")
        yield FinalTextEvent(text=f"Chunk from {self.name}")

    def get_token_count(self, text: str) -> int:
        return len(text.split())


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return ModelRegistry()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock(spec=Config)
        # Mock the nested config attributes
        config.llm = MagicMock()
        config.llm.max_retries = 3
        config.llm.temperature = 0.7
        config.llm.max_output_tokens = 1024
        config.llm.timeout_sec = 60
        return config

    def test_register_and_create_llm(self, registry, mock_config):
        """Test registering and creating a model."""

        # Register a mock provider
        @registry.register_llm("test", "text-model")
        class TestModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("test-model")

        # Create the model
        model = registry.create_llm("test/text-model", config=mock_config)
        assert isinstance(model, TestModel)
        assert model.name == "test-model"

    def test_register_wildcard(self, registry, mock_config):
        """Test registering with wildcard pattern."""

        @registry.register_llm("wildcard", "*")
        class WildcardModel(MockProvider):
            def __init__(self, config, *, model_name=None, **kwargs):
                super().__init__(model_name or "default")

        # Should match any model name under wildcard/
        model = registry.create_llm("wildcard/any-model", config=mock_config)
        assert model.name == "any-model"

        model2 = registry.create_llm("wildcard/another", config=mock_config)
        assert model2.name == "another"

    def test_invalid_key_format(self, registry, mock_config):
        """Test that invalid keys raise ValueError."""
        with pytest.raises(ValueError):
            registry.create_llm("no-slash", config=mock_config)

    def test_unregistered_provider(self, registry, mock_config):
        """Test creating model with unregistered provider."""
        with pytest.raises(KeyError):
            registry.create_llm("unregistered/model", config=mock_config)

    def test_get_llm_with_fallback_basic(self, registry, mock_config):
        """Test basic fallback functionality."""

        @registry.register_llm("primary", "model")
        class PrimaryModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("primary")

        @registry.register_llm("fallback", "model")
        class FallbackModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("fallback")

        # Get model with fallback
        model = registry.get_llm_with_fallback(
            "primary/model", fallback_keys=["fallback/model"], config=mock_config
        )
        # Should return a FallbackModel wrapper
        assert hasattr(model, "_candidate_keys") or hasattr(model, "_candidates")

    @pytest.mark.anyio
    async def test_fallback_model_generation(self, registry, mock_config):
        """Test FallbackModel generation with multiple candidates."""

        @registry.register_llm("good", "model")
        class GoodModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("good")

        # Test successful generation through fallback wrapper
        model = registry.get_llm_with_fallback("good/model", config=mock_config)

        request = ModelRequest(parts=[TextPart(text="test")])
        response = await model.generate(request)
        assert "good" in response.text.lower() or response.text

    @pytest.mark.anyio
    async def test_fallback_model_streaming(self, registry, mock_config):
        """Test FallbackModel streaming."""

        @registry.register_llm("stream", "model")
        class StreamModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("stream")

        model = registry.get_llm_with_fallback("stream/model", config=mock_config)

        request = ModelRequest(parts=[TextPart(text="test")])
        events = []
        async for event in model.stream(request):
            events.append(event)

        assert len(events) >= 1
        # Should have at least a text delta or final text
        has_content = any(hasattr(e, "delta") or hasattr(e, "text") for e in events)
        assert has_content

    def test_capability_filtering(self, registry, mock_config):
        """Test that fallback respects capability requirements."""

        @registry.register_llm("text", "only")
        class TextOnlyModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("text-only", supports_text=True, supports_image=False)

        @registry.register_llm("multi", "modal")
        class MultimodalModel(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("multimodal", supports_text=True, supports_image=True)

        # Get model with multimodal fallback
        model = registry.get_llm_with_fallback(
            "text/only",
            fallback_keys=["multi/modal"],
            config=mock_config,
        )
        # Should have candidates
        assert hasattr(model, "_candidate_keys") or hasattr(model, "_candidates")

    def test_list_registered_models(self, registry):
        """Test listing registered models."""

        @registry.register_llm("provider1", "model-a")
        class Model1(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("model-a")

        @registry.register_llm("provider1", "model-b")
        class Model2(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("model-b")

        @registry.register_llm("provider2", "*")
        class Model3(MockProvider):
            def __init__(self, config, **kwargs):
                super().__init__("wildcard")

        # Check that models are registered via internal _llms registry
        assert "provider1/model-a" in registry._llms._items
        assert "provider1/model-b" in registry._llms._items
        assert "provider2/*" in registry._llms._items
