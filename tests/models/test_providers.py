"""Tests for model provider implementations."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock modules that might not be installed
sys.modules["langchain_google_vertexai"] = MagicMock()
sys.modules["langchain_openai"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

from contextrouter.modules.models.llm.huggingface import HuggingFaceLLM  # noqa: E402
from contextrouter.modules.models.llm.openai import OpenAILLM  # noqa: E402
from contextrouter.modules.models.llm.vertex import VertexLLM  # noqa: E402
from contextrouter.modules.models.types import (  # noqa: E402
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    TextPart,
)


class TestVertexLLM:
    """Test VertexLLM provider."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for Vertex."""
        config = MagicMock()
        config.vertex.project_id = "test-project"
        config.vertex.location = "us-central1"
        config.llm.temperature = 0.7
        config.llm.max_output_tokens = 1024
        config.llm.timeout_sec = 60
        config.llm.max_retries = 3
        return config

    def test_initialization(self, mock_config):
        """Test VertexLLM initialization."""
        with (
            patch("google.auth.default", return_value=(MagicMock(), "test-project")),
            patch("langchain_google_vertexai.ChatVertexAI") as mock_vertex,
        ):
            model = VertexLLM(mock_config, model_name="gemini-2.5-flash")

            assert isinstance(model.capabilities, ModelCapabilities)
            assert model.capabilities.supports_text is True
            assert model.capabilities.supports_image is True  # Gemini supports images
            assert model.capabilities.supports_audio is True  # Gemini supports audio

            mock_vertex.assert_called_once()

    @pytest.mark.anyio
    async def test_generate_text_only(self, mock_config):
        """Test text-only generation."""
        with (
            patch("google.auth.default", return_value=(MagicMock(), "test-project")),
            patch("langchain_google_vertexai.ChatVertexAI") as mock_vertex_class,
        ):
            # Mock the LangChain model
            mock_model = MagicMock()
            mock_model.model_name = "gemini-2.5-flash"
            mock_message = MagicMock()
            mock_message.content = "Generated response"

            async def _mock_ainvoke(*args, **kwargs):
                return mock_message

            # We need to mock the result of bind()
            mock_model.bind.return_value.ainvoke = _mock_ainvoke
            mock_vertex_class.return_value = mock_model

            model = VertexLLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="Hello world")], temperature=0.5)

            response = await model.generate(request)

            assert isinstance(response, ModelResponse)
            assert response.text == "Generated response"
            assert isinstance(response.raw_provider, ProviderInfo)

    @pytest.mark.anyio
    async def test_stream_text_only(self, mock_config):
        """Test text-only streaming."""
        with (
            patch("google.auth.default", return_value=(MagicMock(), "test-project")),
            patch("langchain_google_vertexai.ChatVertexAI") as mock_vertex_class,
        ):
            mock_model = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.content = "chunk"

            async def _mock_astream(*args, **kwargs):
                yield mock_chunk

            # We need to mock the result of bind()
            mock_model.bind.return_value.astream = _mock_astream
            mock_vertex_class.return_value = mock_model

            model = VertexLLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="Hello world")], temperature=0.5)

            events = []
            async for event in model.stream(request):
                events.append(event)

            assert len(events) >= 1
            # One text_delta and one final_text
            assert any(e.event_type == "text_delta" for e in events)

    def test_token_count(self, mock_config):
        """Test token counting."""
        with (
            patch("google.auth.default", return_value=(MagicMock(), "test-project")),
            patch("langchain_google_vertexai.ChatVertexAI") as mock_vertex_class,
        ):
            mock_model = MagicMock()
            mock_model.get_num_tokens.return_value = 42
            mock_vertex_class.return_value = mock_model

            model = VertexLLM(mock_config)
            count = model.get_token_count("hello world")
            assert count == 42
            mock_model.get_num_tokens.assert_called_once_with("hello world")

    def test_missing_config(self):
        """Test error when config is missing required fields."""
        config = MagicMock()
        config.vertex.project_id = None
        config.vertex.location = "us-central1"

        with pytest.raises(ValueError, match="requires vertex.project_id"):
            VertexLLM(config)


class TestOpenAILLM:
    """Test OpenAI provider."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for OpenAI."""
        return MagicMock()

    def test_initialization(self, mock_config):
        """Test OpenAILLM initialization."""
        with patch("langchain_openai.ChatOpenAI"):
            model = OpenAILLM(mock_config, model_name="gpt-5.1")

            assert isinstance(model.capabilities, ModelCapabilities)
            assert model.capabilities.supports_text is True
            assert model.capabilities.supports_image is True
            assert model.capabilities.supports_audio is True

    @pytest.mark.anyio
    async def test_generate(self, mock_config):
        """Test generate."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_model = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "OpenAI response"

            async def _mock_ainvoke(*args, **kwargs):
                return mock_message

            mock_model.bind.return_value.ainvoke = _mock_ainvoke
            mock_chat.return_value = mock_model

            model = OpenAILLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="test")])
            response = await model.generate(request)
            assert response.text == "OpenAI response"

    @pytest.mark.anyio
    async def test_stream(self, mock_config):
        """Test stream."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_model = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.content = "chunk"

            async def _mock_astream(*args, **kwargs):
                yield mock_chunk

            mock_model.bind.return_value.astream = _mock_astream
            mock_chat.return_value = mock_model

            model = OpenAILLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="test")])
            events = []
            async for event in model.stream(request):
                events.append(event)
            assert len(events) >= 1


class TestHuggingFaceLLM:
    """Test HuggingFace provider."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for HuggingFace."""
        return MagicMock()

    def test_initialization(self, mock_config):
        """Test HuggingFaceLLM initialization."""
        model = HuggingFaceLLM(mock_config, model_name="distilgpt2")

        assert isinstance(model.capabilities, ModelCapabilities)
        assert model.capabilities.supports_text is True
        assert model.capabilities.supports_image is False
        assert model.capabilities.supports_audio is False

    @pytest.mark.anyio
    async def test_generate_requires_transformers(self, mock_config):
        """Test that generate fails when model loading fails."""
        model = HuggingFaceLLM(mock_config)
        request = ModelRequest(parts=[TextPart(text="test")])

        # Mock failure in _ensure_model_loaded
        with patch.object(model, "_ensure_model_loaded", side_effect=RuntimeError("Load failed")):
            with pytest.raises(RuntimeError, match="Load failed"):
                await model.generate(request)

    def test_token_count_fallback(self, mock_config):
        """Test token count fallback when transformers not available."""
        model = HuggingFaceLLM(mock_config)
        # Ensure _ensure_model_loaded is a no-op or fails but is caught
        with patch.object(model, "_ensure_model_loaded", side_effect=Exception("No transformers")):
            count = model.get_token_count("hello world test")
            assert count == 3


class TestProviderCapabilities:
    """Test provider capability declarations."""

    def test_vertex_capabilities_by_model(self):
        """Test that Vertex capabilities vary by model."""
        config = MagicMock()
        config.vertex.project_id = "test"
        config.vertex.location = "us-central1"

        with (
            patch("google.auth.default", return_value=(MagicMock(), "test")),
            patch("langchain_google_vertexai.ChatVertexAI"),
        ):
            # Test Gemini 2.5 (multimodal)
            model_25 = VertexLLM(config, model_name="gemini-2.5-flash")
            assert model_25.capabilities.supports_image is True
            assert model_25.capabilities.supports_audio is True

    def test_openai_capabilities_by_model(self):
        """Test that OpenAI capabilities."""
        config = MagicMock()

        with patch("langchain_openai.ChatOpenAI"):
            model_51 = OpenAILLM(config, model_name="gpt-5.1")
            assert model_51.capabilities.supports_text is True
            assert model_51.capabilities.supports_image is True
