"""Tests for model provider implementations.

OpenAI and HuggingFace tests mock their dependencies via sys.modules.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock LLM provider modules that are optional dependencies
sys.modules["langchain_google_vertexai"] = MagicMock()
sys.modules["langchain_openai"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

from contextunity.router.modules.models.llm.huggingface import HuggingFaceLLM  # noqa: E402
from contextunity.router.modules.models.llm.openai import OpenAILLM  # noqa: E402
from contextunity.router.modules.models.types import ModelRequest, TextPart  # noqa: E402


class TestOpenAILLM:
    """Test OpenAI provider — mocked, no external deps needed."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for OpenAI."""
        return MagicMock()

    @pytest.mark.anyio
    async def test_generate(self, mock_config):
        """Test generate."""
        model = OpenAILLM(mock_config)
        mock_resp = MagicMock()
        mock_resp.usage = None
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "OpenAI response"

        from unittest.mock import AsyncMock

        model._openai_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        request = ModelRequest(parts=[TextPart(text="test")])
        response = await model.generate(request)
        assert response.text == "OpenAI response"

    @pytest.mark.anyio
    async def test_gpt5_omits_null_max_tokens(self, mock_config):
        """GPT-5 must not send max_tokens=null when using max_completion_tokens."""
        from unittest.mock import AsyncMock

        model = OpenAILLM(mock_config, model_name="gpt-5-mini")
        mock_resp = MagicMock()
        mock_resp.usage = None
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        create_mock = AsyncMock(return_value=mock_resp)
        model._openai_client.chat.completions.create = create_mock

        await model.generate(ModelRequest(parts=[TextPart(text="test")]))

        kwargs = create_mock.await_args.kwargs
        assert "max_tokens" not in kwargs

    @pytest.mark.anyio
    async def test_json_object_from_model_request(self, mock_config):
        """Planner nodes set response_format on ModelRequest — must reach the API."""
        from unittest.mock import AsyncMock, MagicMock

        model = OpenAILLM(mock_config, model_name="gpt-5-mini")
        mock_resp = MagicMock()
        mock_resp.usage = None
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"ok": true}'
        create_mock = AsyncMock(return_value=mock_resp)
        model._openai_client = MagicMock()
        model._openai_client.chat = MagicMock()
        model._openai_client.chat.completions = MagicMock()
        model._openai_client.chat.completions.create = create_mock

        request = ModelRequest(
            parts=[TextPart(text='Return {"status":"ok"} as json')],
            response_format="json_object",
        )
        await model.generate(request)

        assert create_mock.await_args.kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.anyio
    async def test_json_object_from_provider_config(self, mock_config):
        """Legacy path: response_format only in provider_config must still reach the API."""
        from unittest.mock import AsyncMock, MagicMock

        model = OpenAILLM(mock_config, model_name="gpt-4o-mini")
        mock_resp = MagicMock()
        mock_resp.usage = None
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"ok": true}'
        create_mock = AsyncMock(return_value=mock_resp)
        model._openai_client = MagicMock()
        model._openai_client.chat = MagicMock()
        model._openai_client.chat.completions = MagicMock()
        model._openai_client.chat.completions.create = create_mock

        request = ModelRequest(
            parts=[TextPart(text='Return {"status":"ok"} as json')],
            provider_config={"response_format": "json_object"},
        )
        await model.generate(request)

        assert create_mock.await_args.kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.anyio
    async def test_generation_params_from_model_request(self, mock_config):
        """Node-level temperature/max_tokens must reach the API when set on ModelRequest."""
        from unittest.mock import AsyncMock, MagicMock

        model = OpenAILLM(mock_config, model_name="gpt-4o-mini")
        mock_resp = MagicMock()
        mock_resp.usage = None
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        create_mock = AsyncMock(return_value=mock_resp)
        model._openai_client = MagicMock()
        model._openai_client.chat = MagicMock()
        model._openai_client.chat.completions = MagicMock()
        model._openai_client.chat.completions.create = create_mock

        request = ModelRequest(
            parts=[TextPart(text="test")],
            temperature=0.1,
            max_output_tokens=256,
        )
        await model.generate(request)

        kwargs = create_mock.await_args.kwargs
        assert kwargs.get("temperature") == 0.1
        assert kwargs.get("max_tokens") == 256

    @pytest.mark.anyio
    async def test_stream(self, mock_config):
        """Test stream."""
        model = OpenAILLM(mock_config)
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "chunk"

        async def _mock_stream_generator():
            yield mock_chunk

        async def _mock_create(*args, **kwargs):
            return _mock_stream_generator()

        model._openai_client.chat.completions.create = _mock_create

        request = ModelRequest(parts=[TextPart(text="test")])
        events = []
        async for event in model.stream(request):
            events.append(event)
        assert len(events) >= 1


class TestHuggingFaceLLM:
    """Test HuggingFace provider — mocked, no external deps needed."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for HuggingFace."""
        return MagicMock()

    @pytest.mark.anyio
    async def test_generate_requires_transformers(self, mock_config):
        """Test that generate fails when model loading fails."""
        model = HuggingFaceLLM(mock_config)
        request = ModelRequest(parts=[TextPart(text="test")])

        with patch.object(model, "_ensure_model_loaded", side_effect=RuntimeError("Load failed")):
            with pytest.raises(RuntimeError, match="Load failed"):
                await model.generate(request)

    def test_token_count_fallback(self, mock_config):
        """Test token count fallback when transformers not available."""
        model = HuggingFaceLLM(mock_config)
        with patch.object(model, "_ensure_model_loaded", side_effect=Exception("No transformers")):
            count = model.get_token_count("hello world test")
            assert count == 3


class TestProviderCapabilities:
    """Test provider capability declarations."""

    def test_openai_capabilities_by_model(self):
        """Test that OpenAI capabilities."""
        config = MagicMock()

        with patch("langchain_openai.ChatOpenAI"):
            model_51 = OpenAILLM(config, model_name="gpt-5.1")
            assert model_51.capabilities.supports_text is True
            assert model_51.capabilities.supports_image is True
