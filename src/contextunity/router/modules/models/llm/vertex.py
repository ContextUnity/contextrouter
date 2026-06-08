"""Vertex AI-backed LLM provider (ported from `contextunity.router.cortex.llm`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, override

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.types import JsonDict, is_json_dict
from langchain_core.messages import BaseMessage, SystemMessage

from contextunity.router.core import RouterConfig
from contextunity.router.modules.models.types import ModelError

from ..base import BaseLLM as BaseModel
from ..boundary_common import (
    resolve_json_object_mode,
    resolve_max_output_tokens,
    resolve_temperature,
)
from ..registry import model_registry
from ..types import (
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
    UsageStats,
    VideoPart,
)
from .types import VertexProviderConfig
from .vertex_boundary import (
    GoogleCredentialsProtocol,
    VertexRawModel,
    load_adc_credentials,
    load_chat_vertex_ai,
    load_credentials_from_file,
    refresh_credentials,
    vertex_ai_message_text,
    vertex_chunk_content_text,
    vertex_response_metadata,
    vertex_usage_metadata_mapping,
)

logger = get_contextunit_logger(__name__)


class _VertexRunnable(Protocol):
    async def ainvoke(self, input: list[BaseMessage]) -> object:
        """Forward ``ainvoke`` to the underlying Vertex model."""
        ...

    def astream(self, input: list[BaseMessage]) -> AsyncIterator[object]:
        """Forward ``astream`` to the underlying Vertex model."""
        ...

    def bind(self, **kwargs: object) -> "_VertexRunnable":
        """Return a copy with bound parameters."""
        ...

    def get_num_tokens(self, text: str) -> int:
        """Count tokens in *text* using the model’s tokenizer."""
        ...

    @property
    def model_name(self) -> str:
        """Underlying Vertex model identifier."""
        ...


class _VertexModelWrapper:
    """Typed adapter from ``VertexRawModel`` to ``_VertexRunnable``."""

    def __init__(self, inner: VertexRawModel) -> None:
        self._inner: VertexRawModel = inner

    async def ainvoke(self, input: list[BaseMessage]) -> object:
        return await self._inner.ainvoke(input)

    def astream(self, input: list[BaseMessage]) -> AsyncIterator[object]:
        return self._inner.astream(input)

    def bind(self, **kwargs: object) -> _VertexRunnable:
        return _wrap_vertex_model(self._inner.bind(**kwargs))

    def get_num_tokens(self, text: str) -> int:
        return self._inner.get_num_tokens(text)

    @property
    def model_name(self) -> str:
        return self._inner.model_name


def _wrap_vertex_model(model: object) -> _VertexRunnable:
    """Adapt a ``ChatVertexAI`` instance to the local ``_VertexRunnable`` protocol."""
    if not isinstance(model, VertexRawModel):
        raise ConfigurationError(
            "langchain-google-vertexai returned an incompatible model instance"
        )
    return _VertexModelWrapper(model)


def _int_token(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _usage_from_mapping(data: JsonDict) -> UsageStats | None:
    inp = _int_token(data.get("input_tokens"))
    out = _int_token(data.get("output_tokens"))
    tot = _int_token(data.get("total_tokens"))
    if inp is None and out is None:
        return None
    stats = UsageStats(
        input_tokens=int(inp or 0),
        output_tokens=int(out or 0),
        total_tokens=int(tot or 0) or int(inp or 0) + int(out or 0),
    )
    return stats


def _usage_from_token_usage(data: JsonDict) -> UsageStats | None:
    prompt = _int_token(data.get("prompt_tokens"))
    completion = _int_token(data.get("completion_tokens"))
    total = _int_token(data.get("total_tokens"))
    if prompt is None:
        return None
    return UsageStats(
        input_tokens=int(prompt or 0),
        output_tokens=int(completion or 0),
        total_tokens=int(total or 0),
    )


@model_registry.register_llm("vertex", "gemini-2.5-flash-lite")
@model_registry.register_llm("vertex", "gemini-2.5-flash")
@model_registry.register_llm("vertex", "gemini-2.5-pro")
class VertexLLM(BaseModel):
    """Vertex Gemini via langchain-google-genai.

    Supports multimodal inputs (text, image, audio) where available.

    IMPORTANT: This class MUST NOT access `os.environ` directly. All configuration
    must come from `Config` (which may be layered from env/TOML by the host).
    """

    def __init__(
        self,
        config: RouterConfig,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        streaming: bool = True,
        **_unused: object,
    ) -> None:
        """Resolve Vertex project/location, load ADC credentials, and instantiate the ``ChatVertexAI`` runnable."""
        _ = _unused
        chosen_model = (model_name or "").strip() or "gemini-2.5-flash"
        super().__init__(provider="vertex", model_name=chosen_model)
        self._cfg: RouterConfig = config
        self._credentials: GoogleCredentialsProtocol | None = None
        self._model: _VertexRunnable

        try:
            if config.vertex.credentials_path:
                logger.debug(
                    "VertexLLM: Loading credentials from %s", config.vertex.credentials_path
                )
                self._credentials = load_credentials_from_file(config.vertex.credentials_path)
            else:
                self._credentials = load_adc_credentials()

            if self._credentials is not None:
                refresh_credentials(self._credentials)
        except Exception as e:  # pragma: no cover
            logger.warning("VertexLLM: Failed to initialize credentials: %s", e)
            self._credentials = None

        project_id = config.vertex.project_id
        location = config.vertex.location
        if not project_id or not location:
            raise ConfigurationError(
                (
                    "VertexLLM requires vertex.project_id and vertex.location in Config. "
                    "Set them in core settings.toml under [vertex], or via env vars "
                    "VERTEX_PROJECT_ID and VERTEX_LOCATION "
                    "(you can put them in `.env`). "
                    "When contextunity.router is embedded as a library, the host may also set "
                    "CU_ROUTER_VERTEX_PROJECT_ID / CU_ROUTER_VERTEX_LOCATION."
                )
            )
        if self._credentials is None:
            raise ConfigurationError(
                (
                    "VertexLLM requires valid Google Application Default Credentials (ADC). "
                    "Set GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json "
                    "or run `gcloud auth application-default login`. "
                    "The credentials must have access to Vertex AI in project "
                    f"{project_id!r}."
                )
            )

        self._capabilities: ModelCapabilities = self._get_capabilities(chosen_model)

        try:
            raw_model = load_chat_vertex_ai(
                model_name=chosen_model,
                project=project_id,
                location=location,
                temperature=(config.llm.temperature if temperature is None else temperature),
                max_output_tokens=(
                    config.llm.max_output_tokens if max_output_tokens is None else max_output_tokens
                ),
                streaming=streaming,
                credentials=self._credentials,
            )
            self._model = _wrap_vertex_model(raw_model)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                (
                    "VertexLLM requires `langchain-google-vertexai`. "
                    "Install it via `uv sync --extra vertex` "
                    "(or `pip install langchain-google-vertexai`)."
                )
            ) from e

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the Vertex AI backend."""
        return self._capabilities

    def _get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Determine capabilities based on the model name."""
        if "gemini-1.5" in model_name or "gemini-2.5" in model_name:
            return ModelCapabilities(
                supports_text=True, supports_image=True, supports_audio=True, supports_video=True
            )
        return ModelCapabilities(
            supports_text=True,
            supports_image=False,
            supports_audio=False,
            supports_video=False,
        )

    @override
    async def _generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Call the Gemini API via langchain-google-genai and return a complete response."""

        if not request.parts:
            raise ModelError("Request must contain at least one part")

        messages = self._build_messages(request)
        model = self._apply_request_params(request)

        msg = await model.ainvoke(messages)
        text_result = vertex_ai_message_text(msg)
        usage = self._extract_usage(msg)

        return ModelResponse(
            text=text_result,
            usage=usage,
            raw_provider=self._get_provider_info(),
        )

    def _extract_usage(self, msg: object) -> UsageStats | None:
        """Extract token usage from the response metadata."""
        try:
            um_obj: object | None = getattr(msg, "usage_metadata", None)
            if um_obj is not None:
                stats = _usage_from_mapping(vertex_usage_metadata_mapping(um_obj))
                if stats is not None:
                    model_name = self._model.model_name
                    _ = stats.estimate_cost(model_name)
                    return stats

            rm_obj = vertex_response_metadata(msg)
            if rm_obj is not None:
                tu_obj = rm_obj.get("token_usage")
                if is_json_dict(tu_obj):
                    stats = _usage_from_token_usage(tu_obj)
                    if stats is not None:
                        model_name = self._model.model_name
                        _ = stats.estimate_cost(model_name)
                        return stats
        except Exception:
            logger.debug("Failed to extract usage from Vertex response", exc_info=True)
        return None

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Return a streaming iterator of Gemini completion deltas."""

        if not request.parts:
            raise ConfigurationError("Request must contain at least one part")

        messages = self._build_messages(request)
        model = self._apply_request_params(request)

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the Gemini API via langchain-google-genai stream."""
            full = ""
            async for chunk in model.astream(messages):
                raw_content: object = getattr(chunk, "content", "")
                c = vertex_chunk_content_text(raw_content)
                if c:
                    full += c
                    yield TextDeltaEvent(delta=c)

            yield FinalTextEvent(text=full)

        return _event_stream()

    def _apply_request_params(self, request: ModelRequest) -> _VertexRunnable:
        """Bind request parameters to the underlying model."""
        pc = VertexProviderConfig.model_validate(request.provider_config)
        bind_kwargs: dict[str, object] = {}
        temperature = resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        _max = resolve_max_output_tokens(
            request_max_output_tokens=request.max_output_tokens,
            provider_max_tokens=pc.get_max_tokens(),
        )
        if _max is not None:
            bind_kwargs["max_output_tokens"] = _max
        if resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        ):
            bind_kwargs["response_mime_type"] = "application/json"
        if request.timeout_sec is not None:
            bind_kwargs["timeout"] = request.timeout_sec
        if request.max_retries is not None:
            bind_kwargs["max_retries"] = request.max_retries

        if not bind_kwargs:
            return self._model
        return self._model.bind(**bind_kwargs)

    def _get_provider_info(self) -> ProviderInfo:
        """Return normalized provider info for observability."""
        model_name = self._model.model_name
        return ProviderInfo(
            provider="vertex",
            model_name=model_name,
            model_key=f"vertex/{model_name}",
        )

    def _build_messages(self, request: ModelRequest) -> list[BaseMessage]:
        """Build Gemini-compatible messages with multimodal support."""
        from langchain_core.messages import HumanMessage

        messages: list[BaseMessage] = []
        if request.system:
            messages.append(SystemMessage(content=request.system))

        has_multimodal = any(
            isinstance(p, (ImagePart, AudioPart, VideoPart)) for p in request.parts
        )

        if has_multimodal:
            content: list[str | dict[str, object]] = []
            for part in request.parts:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    if part.data_b64:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{part.mime};base64,{part.data_b64}"},
                            }
                        )
                    elif part.uri:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.uri},
                            }
                        )
                elif isinstance(part, AudioPart):
                    if part.data_b64:
                        content.append(
                            {
                                "type": "media",
                                "media": {"url": f"data:{part.mime};base64,{part.data_b64}"},
                            }
                        )
                    elif part.uri:
                        content.append(
                            {
                                "type": "media",
                                "media": {"url": part.uri},
                            }
                        )
                elif isinstance(part, VideoPart):
                    if part.data_b64:
                        content.append(
                            {
                                "type": "media",
                                "media": {"url": f"data:{part.mime};base64,{part.data_b64}"},
                            }
                        )
                    elif part.uri:
                        content.append(
                            {
                                "type": "media",
                                "media": {"url": part.uri},
                            }
                        )
            messages.append(HumanMessage(content=content))
        else:
            text_parts = [p.text for p in request.parts if isinstance(p, TextPart)]
            if not text_parts:
                raise ModelError("Request must contain at least one text part")
            prompt = "".join(text_parts)
            messages.append(HumanMessage(content=prompt))

        return messages

    @override
    def get_token_count(self, text: str) -> int:
        """Count tokens using the underlying model tokenizer."""
        if not text:
            return 0
        try:
            return self._model.get_num_tokens(text)
        except (AttributeError, TypeError, ValueError):
            return max(1, len(text) // 4)


__all__ = ["VertexLLM"]
