"""Tests for BaseModel interface and core model functionality."""

import asyncio

from contextrouter.modules.models.base import BaseModel
from contextrouter.modules.models.types import (
    AudioPart,
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


class MockModel(BaseModel):
    """Mock model implementation for testing."""

    def __init__(
        self, supports_text: bool = True, supports_image: bool = False, supports_audio: bool = False
    ):
        self._capabilities = ModelCapabilities(
            supports_text=supports_text,
            supports_image=supports_image,
            supports_audio=supports_audio,
        )
        self.generate_call_count = 0
        self.stream_call_count = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(self, request: ModelRequest, *, token=None) -> ModelResponse:
        self.generate_call_count += 1
        return ModelResponse(
            text="mock response",
            raw_provider=ProviderInfo(
                provider="mock", model_name="mock-model", model_key="mock/mock-model"
            ),
        )

    async def stream(self, request: ModelRequest, *, token=None):
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

    def test_text_only_request(self):
        """Test creating a text-only request."""
        request = ModelRequest(
            parts=[TextPart(text="Hello world")],
            temperature=0.7,
            max_output_tokens=100,
        )

        assert len(request.parts) == 1
        assert isinstance(request.parts[0], TextPart)
        assert request.parts[0].text == "Hello world"
        assert request.temperature == 0.7
        assert request.max_output_tokens == 100

    def test_multimodal_request(self):
        """Test creating a multimodal request."""
        request = ModelRequest(
            parts=[
                TextPart(text="Describe this image:"),
                ImagePart(
                    mime="image/jpeg", data_b64="fake_data", uri="http://example.com/image.jpg"
                ),
                AudioPart(mime="audio/wav", data_b64="fake_audio", sample_rate_hz=44100),
            ],
            temperature=0.5,
        )

        assert len(request.parts) == 3
        assert isinstance(request.parts[0], TextPart)
        assert isinstance(request.parts[1], ImagePart)
        assert isinstance(request.parts[2], AudioPart)

        # Check image part
        img_part = request.parts[1]
        assert img_part.mime == "image/jpeg"
        assert img_part.data_b64 == "fake_data"
        assert img_part.uri == "http://example.com/image.jpg"

        # Check audio part
        audio_part = request.parts[2]
        assert audio_part.mime == "audio/wav"
        assert audio_part.sample_rate_hz == 44100

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


class TestModelResponse:
    """Test ModelResponse functionality."""

    def test_response_creation(self):
        """Test creating a model response."""
        response = ModelResponse(
            text="Generated response",
            raw_provider=ProviderInfo(
                provider="test", model_name="test-model", model_key="test/test-model"
            ),
        )
        assert response.text == "Generated response"
        assert response.raw_provider.provider == "test"
        assert response.usage is None

    def test_response_with_usage(self):
        """Test response with usage statistics."""
        from contextrouter.modules.models.types import UsageStats

        response = ModelResponse(
            text="Response with usage",
            raw_provider=ProviderInfo(
                provider="test", model_name="test-model", model_key="test/test-model"
            ),
            usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30


class TestModelStreamEvent:
    """Test ModelStreamEvent functionality."""

    def test_stream_event_creation(self):
        """Test creating stream events."""
        delta_event = TextDeltaEvent(delta="Hello")
        assert delta_event.event_type == "text_delta"
        assert delta_event.delta == "Hello"

        final_event = FinalTextEvent(text="Hello World")
        assert final_event.event_type == "final_text"
        assert final_event.text == "Hello World"


class TestModelParts:
    """Test individual model input parts."""

    def test_text_part(self):
        """Test TextPart creation and validation."""
        part = TextPart(text="Hello world")
        assert part.kind == "text"
        assert part.text == "Hello world"

    def test_image_part_b64(self):
        """Test ImagePart with base64 data."""
        part = ImagePart(
            mime="image/png",
            data_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        )
        assert part.kind == "image"
        assert part.mime == "image/png"
        assert part.data_b64 is not None
        assert part.uri is None

    def test_image_part_uri(self):
        """Test ImagePart with URI."""
        part = ImagePart(mime="image/jpeg", uri="https://example.com/image.jpg")
        assert part.uri == "https://example.com/image.jpg"
        assert part.data_b64 is None

    def test_audio_part(self):
        """Test AudioPart creation."""
        part = AudioPart(mime="audio/mpeg", data_b64="fake_audio_data", sample_rate_hz=44100)
        assert part.kind == "audio"
        assert part.mime == "audio/mpeg"
        assert part.sample_rate_hz == 44100

    def test_audio_part_uri(self):
        """Test AudioPart with URI and no sample rate."""
        part = AudioPart(mime="audio/wav", uri="https://example.com/audio.wav")
        assert part.sample_rate_hz is None
        assert part.uri == "https://example.com/audio.wav"

    def test_video_part(self):
        """Test VideoPart creation."""
        from contextrouter.modules.models.types import VideoPart

        part = VideoPart(mime="video/mp4", uri="gs://bucket/video.mp4")
        assert part.kind == "video"
        assert part.mime == "video/mp4"
        assert part.uri == "gs://bucket/video.mp4"
