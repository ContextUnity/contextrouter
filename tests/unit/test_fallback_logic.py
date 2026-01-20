"""Tests for multimodal fallback logic."""

import pytest

from contextrouter.core.config import Config
from contextrouter.modules.models.base import BaseModel
from contextrouter.modules.models.registry import FallbackModel
from contextrouter.modules.models.types import (
    ImagePart,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    TextPart,
)


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(
        self, capabilities: ModelCapabilities, should_fail: bool = False, response_text: str = "OK"
    ):
        self._capabilities = capabilities
        self._should_fail = should_fail
        self._response_text = response_text

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(self, request, *, token=None):
        if self._should_fail:
            raise Exception("Mock failure")
        return ModelResponse(
            text=self._response_text,
            raw_provider=ProviderInfo(provider="mock", model_name="test", model_key="mock/test"),
        )

    async def stream(self, request, *, token=None):
        if self._should_fail:
            raise Exception("Mock failure")
        from contextrouter.modules.models.types import FinalTextEvent, TextDeltaEvent

        yield TextDeltaEvent(delta=self._response_text)
        yield FinalTextEvent(text=self._response_text)

    def get_token_count(self, text: str) -> int:
        return len(text.split())


class TestFallbackLogic:
    """Test capability-based fallback."""

    def test_capability_filtering(self):
        """Test that only compatible models are selected."""
        config = Config()

        # Create mock models with different capabilities
        text_only = MockModel(ModelCapabilities(supports_text=True, supports_image=False))
        multimodal = MockModel(ModelCapabilities(supports_text=True, supports_image=True))
        failing = MockModel(
            ModelCapabilities(supports_text=True, supports_image=True), should_fail=True
        )

        candidates = [("text-only", text_only), ("multimodal", multimodal), ("failing", failing)]

        fallback = FallbackModel(None, ["text-only", "multimodal", "failing"], "fallback", config)
        fallback._candidates = candidates

        # Text-only request should filter to both text-capable models
        text_request = ModelRequest(parts=[TextPart(text="Hello")])
        filtered = fallback._filter_candidates(text_request.required_modalities())
        assert len(filtered) == 3  # All support text

        # Image request should filter to multimodal only
        image_request = ModelRequest(
            parts=[TextPart(text="Hi"), ImagePart(mime="image/png", uri="gcs://test")]
        )
        filtered = fallback._filter_candidates(image_request.required_modalities())
        assert len(filtered) == 2  # multimodal and failing
        assert filtered[0][0] == "multimodal"
        assert filtered[1][0] == "failing"

    def test_no_compatible_models_error(self):
        """Test error when no models support required modalities."""
        config = Config()

        text_only = MockModel(ModelCapabilities(supports_text=True, supports_image=False))
        candidates = [("text-only", text_only)]

        fallback = FallbackModel(None, ["text-only"], "fallback", config)
        fallback._candidates = candidates

        image_request = ModelRequest(
            parts=[TextPart(text="Hi"), ImagePart(mime="image/png", uri="gcs://test")]
        )

        with pytest.raises(Exception):  # Should raise ModelCapabilityError
            fallback._filter_candidates(image_request.required_modalities())

    @pytest.mark.anyio
    async def test_sequential_fallback_success(self):
        """Test sequential fallback with first model succeeding."""
        config = Config()

        success_model = MockModel(ModelCapabilities(supports_text=True), response_text="Success")
        candidates = [("success", success_model)]

        fallback = FallbackModel(None, ["success"], "fallback", config)
        fallback._candidates = candidates

        request = ModelRequest(parts=[TextPart(text="Test")])
        response = await fallback.generate(request)

        assert response.text == "Success"

    @pytest.mark.anyio
    async def test_sequential_fallback_with_retry(self):
        """Test sequential fallback with first model failing."""
        config = Config()

        failing_model = MockModel(ModelCapabilities(supports_text=True), should_fail=True)
        success_model = MockModel(
            ModelCapabilities(supports_text=True), response_text="Fallback success"
        )

        candidates = [("failing", failing_model), ("success", success_model)]

        fallback = FallbackModel(None, ["failing", "success"], "fallback", config)
        fallback._candidates = candidates

        request = ModelRequest(parts=[TextPart(text="Test")])
        response = await fallback.generate(request)

        assert response.text == "Fallback success"
