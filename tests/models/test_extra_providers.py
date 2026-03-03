"""Tests for new model provider implementations."""

from unittest.mock import MagicMock, patch

import pytest

# Mock modules as needed via tests, do not mock globally.
from contextrouter.modules.models.llm.groq import GroqLLM  # noqa: E402
from contextrouter.modules.models.llm.hf_hub import HuggingFaceHubLLM  # noqa: E402
from contextrouter.modules.models.llm.inception import InceptionLLM  # noqa: E402
from contextrouter.modules.models.llm.runpod import RunPodLLM  # noqa: E402
from contextrouter.modules.models.types import (  # noqa: E402
    ModelCapabilities,
    ModelRequest,
    TextPart,
)


class TestExtraProviders:
    """Test Groq, RunPod, HF Hub, and Inception providers."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.groq.api_key = "test-groq-key"
        config.groq.base_url = "https://api.groq.com/openai/v1"
        config.runpod.api_key = "test-runpod-key"
        config.runpod.base_url = "https://api.runpod.ai/v2/test/openai/v1"
        config.hf_hub.api_key = "test-hf-key"
        config.hf_hub.base_url = "https://api-inference.huggingface.co/v1"
        config.inception.api_key = "test-inception-key"
        config.inception.base_url = "https://api.inceptionlabs.ai/v1"
        config.inception.reasoning_effort = "medium"
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
        from unittest.mock import AsyncMock

        mock_choice = MagicMock()
        mock_choice.message.content = "Groq response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        model = GroqLLM(mock_config)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._client = mock_client

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

            from unittest.mock import AsyncMock

            mock_client.chat_completion = AsyncMock(return_value=mock_resp)

            model = HuggingFaceHubLLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="Hello")])
            resp = await model.generate(request)

            assert resp.text == "HF Hub response"
            assert resp.raw_provider.provider == "hf-hub"

    def test_inception_initialization(self, mock_config):
        model = InceptionLLM(mock_config, model_name="mercury-2")
        assert isinstance(model.capabilities, ModelCapabilities)
        assert model.capabilities.supports_text is True
        assert model.capabilities.supports_image is False
        assert model.capabilities.supports_audio is False

    def test_inception_default_model(self, mock_config):
        model = InceptionLLM(mock_config)
        assert model._model_name == "mercury-2"

    @pytest.mark.anyio
    async def test_inception_generate(self, mock_config):
        from unittest.mock import AsyncMock

        mock_choice = MagicMock()
        mock_choice.message.content = "Mercury-2 response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        model = InceptionLLM(mock_config)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._client = mock_client

        request = ModelRequest(parts=[TextPart(text="Hello")])
        resp = await model.generate(request)

        assert resp.text == "Mercury-2 response"
        assert resp.raw_provider.provider == "inception"
        assert resp.raw_provider.model_name == "mercury-2"
