"""Tests for new model provider implementations."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock modules that might not be installed
sys.modules["langchain_openai"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()

from contextrouter.modules.models.llm.groq import GroqLLM  # noqa: E402
from contextrouter.modules.models.llm.hf_hub import HuggingFaceHubLLM  # noqa: E402
from contextrouter.modules.models.llm.runpod import RunPodLLM  # noqa: E402
from contextrouter.modules.models.types import (  # noqa: E402
    ModelCapabilities,
    ModelRequest,
    TextPart,
)


class TestExtraProviders:
    """Test Groq, RunPod, and HF Hub providers."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.groq.api_key = "test-groq-key"
        config.groq.base_url = "https://api.groq.com/openai/v1"
        config.runpod.api_key = "test-runpod-key"
        config.runpod.base_url = "https://api.runpod.ai/v2/test/openai/v1"
        config.hf_hub.api_key = "test-hf-key"
        config.hf_hub.base_url = "https://api-inference.huggingface.co/v1"
        config.llm.max_retries = 3
        return config

    def test_groq_initialization(self, mock_config):
        model = GroqLLM(mock_config, model_name="llama-3.3-70b-versatile")
        assert isinstance(model.capabilities, ModelCapabilities)
        assert model.capabilities.supports_text is True
        assert model.capabilities.supports_image is True

    def test_runpod_initialization(self, mock_config):
        model = RunPodLLM(mock_config, model_name="llama3-8b")
        assert isinstance(model.capabilities, ModelCapabilities)
        assert model.capabilities.supports_text is True
        assert model.capabilities.supports_image is True

    def test_hf_hub_initialization(self, mock_config):
        model = HuggingFaceHubLLM(mock_config, model_name="mistralai/Mistral-7B-Instruct-v0.2")
        assert isinstance(model.capabilities, ModelCapabilities)
        assert model.capabilities.supports_text is True
        assert model.capabilities.supports_audio is False

    @pytest.mark.anyio
    async def test_groq_generate(self, mock_config):
        # We need to mock the instance created inside GroqLLM
        mock_model = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Groq response"

        async def _mock_ainvoke(*args, **kwargs):
            return mock_msg

        mock_model.ainvoke = _mock_ainvoke

        # Patch the ChatOpenAI class that was mocked in sys.modules
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value.bind.return_value = mock_model

            model = GroqLLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="Hello")])
            resp = await model.generate(request)

            assert resp.text == "Groq response"
            assert resp.raw_provider.provider == "groq"

    @pytest.mark.anyio
    async def test_hf_hub_generate_text(self, mock_config):
        # Patch the AsyncInferenceClient class that was mocked in sys.modules
        with patch("huggingface_hub.AsyncInferenceClient") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "HF Hub response"

            async def _mock_chat(*args, **kwargs):
                return mock_resp

            mock_client.chat_completion = _mock_chat

            model = HuggingFaceHubLLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="Hello")])
            resp = await model.generate(request)

            assert resp.text == "HF Hub response"
            assert resp.raw_provider.provider == "hf-hub"
