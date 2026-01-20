"""Tests for multimodal model types and contracts."""

import pytest
from pydantic import ValidationError

from contextrouter.modules.models.types import (
    AudioPart,
    ImagePart,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    TextPart,
    UsageStats,
)


class TestModelParts:
    """Test discriminated union for model parts."""

    def test_text_part_creation(self):
        part = TextPart(text="Hello world")
        assert part.kind == "text"
        assert part.text == "Hello world"

    def test_image_part_creation(self):
        part = ImagePart(
            mime="image/jpeg", data_b64="base64data", uri="https://example.com/image.jpg"
        )
        assert part.kind == "image"
        assert part.mime == "image/jpeg"
        assert part.data_b64 == "base64data"
        assert part.uri == "https://example.com/image.jpg"

    def test_audio_part_creation(self):
        part = AudioPart(mime="audio/wav", data_b64="audio_data", sample_rate_hz=44100)
        assert part.kind == "audio"
        assert part.mime == "audio/wav"
        assert part.sample_rate_hz == 44100

    def test_invalid_part_fails(self):
        with pytest.raises(ValidationError):
            # Missing required fields
            TextPart()


class TestModelCapabilities:
    """Test capability matching logic."""

    def test_text_only_capabilities(self):
        caps = ModelCapabilities(supports_text=True, supports_image=False, supports_audio=False)

        assert caps.supports({"text"})
        assert not caps.supports({"image"})
        assert not caps.supports({"audio"})
        assert not caps.supports({"text", "image"})

    def test_multimodal_capabilities(self):
        caps = ModelCapabilities(supports_text=True, supports_image=True, supports_audio=True)

        assert caps.supports({"text"})
        assert caps.supports({"image"})
        assert caps.supports({"audio"})
        assert caps.supports({"text", "image"})
        assert caps.supports({"text", "image", "audio"})

    def test_empty_capabilities(self):
        caps = ModelCapabilities()
        # Default is text=True, others=False
        assert caps.supports({"text"})
        assert not caps.supports({"image"})


class TestModelRequest:
    """Test model request validation."""

    def test_text_only_request(self):
        request = ModelRequest(
            parts=[TextPart(text="Hello")], temperature=0.7, max_output_tokens=100
        )
        assert len(request.parts) == 1
        assert request.required_modalities() == {"text"}
        assert request.temperature == 0.7

    def test_multimodal_request(self):
        request = ModelRequest(
            parts=[
                TextPart(text="What's in this image?"),
                ImagePart(mime="image/jpeg", data_b64="data"),
            ]
        )
        assert len(request.parts) == 2
        assert request.required_modalities() == {"text", "image"}

    def test_empty_request_fails(self):
        with pytest.raises(ValidationError):
            ModelRequest(parts=[])


class TestModelResponse:
    """Test model response validation."""

    def test_basic_response(self):
        response = ModelResponse(
            text="Hello back!",
            raw_provider=ProviderInfo(
                provider="vertex",
                model_name="gemini-2.5-flash",
                model_key="vertex/gemini-2.5-flash",
            ),
        )
        assert response.text == "Hello back!"
        assert response.raw_provider.provider == "vertex"

    def test_response_with_usage(self):
        usage = UsageStats(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        )
        response = ModelResponse(
            text="Response",
            usage=usage,
            raw_provider=ProviderInfo(
                provider="openai", model_name="gpt-4", model_key="openai/gpt-4"
            ),
        )
        assert response.usage.total_tokens == 15
        assert response.usage.total_cost == 0.003
