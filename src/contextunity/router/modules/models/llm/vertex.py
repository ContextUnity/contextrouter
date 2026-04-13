"""Vertex AI-backed LLM provider (ported from `contextunity.router.cortex.llm`)."""

from __future__ import annotations

from typing import AsyncIterator

from contextunity.core import get_contextunit_logger
from contextunity.core.tokens import ContextToken
from langchain_core.messages import SystemMessage

from contextunity.router.core import Config

from ..base import BaseModel
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

logger = get_contextunit_logger(__name__)


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
        config: Config,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        streaming: bool = True,
        **_: object,
    ) -> None:
        self._cfg = config
        self._credentials = None

        # Initialize Google credentials.
        try:
            import google.auth
            import google.auth.transport.requests
            from google.oauth2 import service_account
        except Exception as e:  # pragma: no cover
            logger.warning("VertexLLM: google-auth not available: %s", e)
            self._credentials = None
        else:
            try:
                # 1. Try explicit path from config
                if config.vertex.credentials_path:
                    logger.debug(
                        "VertexLLM: Loading credentials from %s", config.vertex.credentials_path
                    )
                    self._credentials = service_account.Credentials.from_service_account_file(
                        config.vertex.credentials_path,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                else:
                    # 2. Fallback to ADC (Application Default Credentials)
                    self._credentials, _project = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )

                if self._credentials and hasattr(self._credentials, "refresh"):
                    self._credentials.refresh(google.auth.transport.requests.Request())
            except Exception as e:  # pragma: no cover
                logger.warning("VertexLLM: Failed to initialize credentials: %s", e)
                self._credentials = None

        chosen_model = (model_name or "").strip() or "gemini-2.5-flash"

        project_id = config.vertex.project_id
        location = config.vertex.location
        if not project_id or not location:
            # Keep error explicit and early (enterprise-friendly).
            raise ValueError(
                "VertexLLM requires vertex.project_id and vertex.location in Config. "
                "Set them in core settings.toml under [vertex], or via env vars "
                "VERTEX_PROJECT_ID and VERTEX_LOCATION "
                "(you can put them in `.env`). "
                "When contextunity.router is embedded as a library, the host may also set "
                "CU_ROUTER_VERTEX_PROJECT_ID / CU_ROUTER_VERTEX_LOCATION."
            )
        if self._credentials is None:
            raise ValueError(
                "VertexLLM requires valid Google Application Default Credentials (ADC). "
                "Set GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json "
                "or run `gcloud auth application-default login`. "
                "The credentials must have access to Vertex AI in project "
                f"{project_id!r}."
            )

        # Determine capabilities based on model
        self._capabilities = self._get_capabilities(chosen_model)

        # Prefer the Vertex-native LangChain integration when available.
        # This avoids ctor-arg drift in langchain-google-genai and keeps auth on ADC.
        try:
            from langchain_google_vertexai import ChatVertexAI  # type: ignore[import-not-found]

            self._model = ChatVertexAI(
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
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "VertexLLM requires `langchain-google-vertexai`. "
                "Install it via `uv sync --extra vertex` (or `pip install langchain-google-vertexai`)."
            ) from e

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Determine capabilities based on model name."""
        # Gemini 1.5 and 2.5 models support multimodal (text, image, audio, video)
        if "gemini-1.5" in model_name or "gemini-2.5" in model_name:
            return ModelCapabilities(
                supports_text=True, supports_image=True, supports_audio=True, supports_video=True
            )
        # Fallback to text-only for older models
        return ModelCapabilities(
            supports_text=True,
            supports_image=False,
            supports_audio=False,
            supports_video=False,
        )

    async def generate(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
    ) -> ModelResponse:
        _ = token
        if not request.parts:
            raise ValueError("Request must contain at least one part")

        # Build multimodal messages
        messages = self._build_messages(request)
        model = self._apply_request_params(request)

        msg = await model.ainvoke(messages)
        content = getattr(msg, "content", "")
        text_result = content if isinstance(content, str) else str(content)

        usage = self._extract_usage(msg)

        return ModelResponse(
            text=text_result,
            usage=usage,
            raw_provider=self._get_provider_info(),
        )

    def _extract_usage(self, msg: object) -> UsageStats | None:
        """Extract token usage from LangChain Gemini response metadata."""
        try:
            # LangChain usage_metadata (works for Vertex AI)
            um = getattr(msg, "usage_metadata", None)
            if um:
                inp = getattr(um, "input_tokens", None) or (
                    um.get("input_tokens") if isinstance(um, dict) else None
                )
                out = getattr(um, "output_tokens", None) or (
                    um.get("output_tokens") if isinstance(um, dict) else None
                )
                tot = getattr(um, "total_tokens", None) or (
                    um.get("total_tokens") if isinstance(um, dict) else None
                )
                if inp is not None or out is not None:
                    stats = UsageStats(
                        input_tokens=int(inp or 0),
                        output_tokens=int(out or 0),
                        total_tokens=int(tot or 0) or int(inp or 0) + int(out or 0),
                    )
                    model_name = getattr(self._model, "model_name", "")
                    stats.estimate_cost(model_name)
                    return stats

            # Fallback: response_metadata.token_usage
            rm = getattr(msg, "response_metadata", None) or {}
            if isinstance(rm, dict):
                tu = rm.get("token_usage", {})
                if isinstance(tu, dict) and tu.get("prompt_tokens") is not None:
                    stats = UsageStats(
                        input_tokens=int(tu.get("prompt_tokens") or 0),
                        output_tokens=int(tu.get("completion_tokens") or 0),
                        total_tokens=int(tu.get("total_tokens") or 0),
                    )
                    model_name = getattr(self._model, "model_name", "")
                    stats.estimate_cost(model_name)
                    return stats
        except Exception:
            logger.debug("Failed to extract usage from Vertex response", exc_info=True)
        return None

    async def stream(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = token
        if not request.parts:
            raise ValueError("Request must contain at least one part")

        # Build multimodal messages
        messages = self._build_messages(request)
        model = self._apply_request_params(request)

        full = ""
        async for chunk in model.astream(messages):
            raw_content = getattr(chunk, "content", "")
            # Handle both string and list content (LangChain can return either)
            if isinstance(raw_content, str):
                c = raw_content
            elif isinstance(raw_content, list):
                # Extract text from list content (multimodal format)
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, str):
                        text_parts.append(part)
                c = "".join(text_parts)
            else:
                # Fallback: convert to string
                c = str(raw_content) if raw_content else ""

            if c:
                full += c
                yield TextDeltaEvent(delta=c)

        yield FinalTextEvent(text=full)

    def _apply_request_params(self, request: ModelRequest) -> object:
        """Bind request parameters to the underlying model."""
        bind_kwargs: dict[str, object] = {}
        if request.temperature is not None:
            bind_kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            bind_kwargs["max_output_tokens"] = request.max_output_tokens
        if request.timeout_sec is not None:
            bind_kwargs["timeout"] = request.timeout_sec
        if request.max_retries is not None:
            bind_kwargs["max_retries"] = request.max_retries

        if not bind_kwargs:
            return self._model
        return self._model.bind(**bind_kwargs)

    def _get_provider_info(self) -> ProviderInfo:
        """Get normalized provider information."""
        model_name = getattr(self._model, "model_name", "unknown")
        return ProviderInfo(
            provider="vertex",
            model_name=model_name,
            model_key=f"vertex/{model_name}",
        )

    def _build_messages(self, request: ModelRequest) -> list[object]:
        """Build Gemini-compatible messages with multimodal support."""
        from langchain_core.messages import HumanMessage

        messages: list[object] = []
        if request.system:
            messages.append(SystemMessage(content=request.system))

        # Check for multimodal parts
        has_multimodal = any(
            isinstance(p, (ImagePart, AudioPart, VideoPart)) for p in request.parts
        )

        if has_multimodal:
            # Build multimodal content for Gemini
            # Gemini uses a content array with different part types
            content: list[dict[str, object]] = []
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
                    # Gemini accepts audio via data URL or GCS URI
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
                    # Gemini accepts video via GCS URI primarily
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
            # Text-only: simple string content
            text_parts = [p.text for p in request.parts if isinstance(p, TextPart)]
            if not text_parts:
                raise ValueError("Request must contain at least one text part")
            prompt = "".join(text_parts)
            messages.append(HumanMessage(content=prompt))

        return messages

    def get_token_count(self, text: str) -> int:
        """Count tokens using the underlying model's tokenizer."""
        if not text:
            return 0
        try:
            return self._model.get_num_tokens(text)
        except (AttributeError, TypeError, ValueError):
            # Fallback to a rough estimate (approx 4 chars per token)
            return max(1, len(text) // 4)


__all__ = ["VertexLLM"]
