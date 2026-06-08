"""Tests for new model provider implementations."""

from unittest.mock import MagicMock, patch

import pytest

# Mock modules as needed via tests, do not mock globally.
from contextunity.router.modules.models.llm.groq import GroqLLM  # noqa: E402
from contextunity.router.modules.models.llm.hf_hub import HuggingFaceHubLLM  # noqa: E402
from contextunity.router.modules.models.llm.inception import InceptionLLM  # noqa: E402
from contextunity.router.modules.models.types import (  # noqa: E402
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

    @pytest.mark.anyio
    async def test_groq_generate(self, mock_config):
        model = GroqLLM(mock_config)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Groq response"

        from unittest.mock import AsyncMock

        model._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        request = ModelRequest(parts=[TextPart(text="Hello")])
        resp = await model.generate(request)

        assert resp.text == "Groq response"
        assert resp.raw_provider.provider == "groq"

    @pytest.mark.anyio
    async def test_hf_hub_generate_text(self, mock_config):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "HF Hub response"

        from unittest.mock import AsyncMock

        mock_client.chat_completion = AsyncMock(return_value=mock_resp)

        with patch(
            "contextunity.router.modules.models.llm.hf_hub.load_async_inference_client",
            return_value=mock_client,
        ):
            model = HuggingFaceHubLLM(mock_config)
            request = ModelRequest(parts=[TextPart(text="Hello")])
            resp = await model.generate(request)

            assert resp.text == "HF Hub response"
            assert resp.raw_provider.provider == "hf-hub"

    @pytest.mark.anyio
    async def test_inception_generate(self, mock_config):
        model = InceptionLLM(mock_config)

        mock_resp = MagicMock()
        mock_resp.usage = None
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Mercury-2 response"

        from unittest.mock import AsyncMock

        model._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        request = ModelRequest(parts=[TextPart(text="Hello")])
        resp = await model.generate(request)

        assert resp.text == "Mercury-2 response"
        assert resp.raw_provider.provider == "inception"
        assert resp.raw_provider.model_name == "mercury-2"
