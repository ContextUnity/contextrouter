"""Tests for BaseModel interface and core model functionality."""

import asyncio

from contextunity.router.modules.models.base import BaseLLM
from contextunity.router.modules.models.types import (
    FinalTextEvent,
    ImagePart,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
    TextPart,
)


class MockModel(BaseLLM):
    """Mock model implementation for testing."""

    def __init__(
        self, supports_text: bool = True, supports_image: bool = False, supports_audio: bool = False
    ):
        super().__init__(provider="mock", model_name="mock-model")
        self._capabilities = ModelCapabilities(
            supports_text=supports_text,
            supports_image=supports_image,
            supports_audio=supports_audio,
        )
        self.generate_call_count = 0
        self.stream_call_count = 0

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        self.generate_call_count += 1
        return ModelResponse(
            text="mock response",
            raw_provider=ProviderInfo(
                provider="mock", model_name="mock-model", model_key="mock/mock-model"
            ),
        )

    async def _stream(self, request: ModelRequest):
        self.stream_call_count += 1
        yield TextDeltaEvent(delta="mock")
        yield FinalTextEvent(text="mock")

    def get_token_count(self, text: str) -> int:
        return len(text.split())


class TestBaseModel:
    """Test the BaseModel abstract interface."""

    def test_abstract_methods(self):
        """Test that BaseModel defines the required abstract methods."""
        model = MockModel()

        # Test capabilities property
        caps = model.capabilities
        assert isinstance(caps, ModelCapabilities)
        assert caps.supports_text is True
        assert caps.supports_image is False
        assert caps.supports_audio is False

        # Test generate method
        request = ModelRequest(parts=[TextPart(text="test")])
        response = asyncio.run(model.generate(request))
        assert isinstance(response, ModelResponse)
        assert response.text == "mock response"
        assert model.generate_call_count == 1
        assert response.raw_provider.provider == "mock"

        # Test stream method
        events: list[ModelStreamEvent] = []

        async def collect_events():
            async for event in model.stream(request):
                events.append(event)

        asyncio.run(collect_events())
        assert len(events) == 2
        assert isinstance(events[0], TextDeltaEvent)
        assert events[0].delta == "mock"
        assert isinstance(events[1], FinalTextEvent)
        assert events[1].text == "mock"
        assert model.stream_call_count == 1

        # Test get_token_count
        count = model.get_token_count("hello world test")
        assert count == 3


class TestModelCapabilities:
    """Test ModelCapabilities functionality."""

    def test_supports_method(self):
        """Test the supports method for modality checking."""
        # Text-only model
        text_caps = ModelCapabilities(
            supports_text=True, supports_image=False, supports_audio=False
        )
        assert text_caps.supports({"text"}) is True
        assert text_caps.supports({"image"}) is False
        assert text_caps.supports({"text", "image"}) is False

        # Multimodal model
        multi_caps = ModelCapabilities(supports_text=True, supports_image=True, supports_audio=True)
        assert multi_caps.supports({"text"}) is True
        assert multi_caps.supports({"image"}) is True
        assert multi_caps.supports({"audio"}) is True
        assert multi_caps.supports({"text", "image"}) is True
        assert multi_caps.supports({"text", "image", "audio"}) is True
        assert multi_caps.supports({"video"}) is False  # Unsupported modality

    def test_supports_video(self):
        """Test video capability."""
        caps = ModelCapabilities(supports_text=True, supports_video=True)
        assert caps.supports({"video"}) is True
        assert caps.supports({"text", "video"}) is True


class TestModelRequest:
    """Test ModelRequest creation and validation."""

    def test_request_validation(self):
        """Test request validation."""
        # Test with system message
        request = ModelRequest(parts=[TextPart(text="Hello")], system="You are a helpful assistant")
        assert request.system == "You are a helpful assistant"

    def test_required_modalities(self):
        """Test extracting required modalities from request."""
        request = ModelRequest(
            parts=[
                TextPart(text="test"),
                ImagePart(mime="image/png", data_b64="data"),
            ]
        )
        modalities = request.required_modalities()
        assert modalities == {"text", "image"}

    def test_to_text_prompt(self):
        """Test converting request to text prompt."""
        request = ModelRequest(
            parts=[TextPart(text="Hello"), TextPart(text="World")],
            system="Be helpful",
        )
        prompt = request.to_text_prompt(include_system=True)
        assert "Be helpful" in prompt
        assert "Hello" in prompt
        assert "World" in prompt


class TestModelParts:
    """Test individual model input parts."""

    def test_video_part(self):
        """Test VideoPart creation."""
        from contextunity.router.modules.models.types import VideoPart

        part = VideoPart(mime="video/mp4", uri="gs://bucket/video.mp4")
        assert part.kind == "video"
        assert part.mime == "video/mp4"
        assert part.uri == "gs://bucket/video.mp4"
