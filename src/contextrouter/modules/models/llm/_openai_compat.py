"""Shared utilities for OpenAI-compatible providers.

This module extracts common patterns used across multiple OpenAI-compatible
providers (OpenAI, Groq, OpenRouter, RunPod, Local) to reduce code duplication.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import ModelRequest, ModelResponse


def build_native_openai_messages(
    request: "ModelRequest", is_reasoning_model: bool = False
) -> list[dict[str, object]]:
    """Build native OpenAI API compatible messages with multimodal support.

    This handles native `dict` message building for direct SDK/HTTP use,
    replacing the legacy LangChain-based `build_openai_messages`.

    Handles:
    - System/Developer messages (based on model reasoning family)
    - Text-only content (simple string)
    - Multimodal content with images (content array format)

    Args:
        request: The model request containing parts and system prompt.
        is_reasoning_model: If True, uses 'developer' role instead of 'system'.

    Returns:
        List of dict message objects ready for OpenAI SDK context arrays.
    """
    from ..types import ImagePart, TextPart

    messages: list[dict[str, object]] = []

    if request.system:
        role = "developer" if is_reasoning_model else "system"
        messages.append({"role": role, "content": request.system})

    has_images = any(isinstance(p, ImagePart) for p in request.parts)

    if has_images:
        content: list[dict[str, object]] = []
        for part in request.parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if part.uri:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": part.uri},
                        }
                    )
                elif part.data_b64:
                    data_url = f"data:{part.mime or 'image/jpeg'};base64,{part.data_b64}"
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
        messages.append({"role": "user", "content": content})
    else:
        # Fallback to pure string processing if no images
        messages.append({"role": "user", "content": request.to_text_prompt()})

    return messages


def build_openai_messages(
    request: "ModelRequest", is_reasoning_model: bool = False
) -> list[object]:
    """Build OpenAI-compatible messages with multimodal support.

    This handles:
    - System messages
    - Text-only content (simple string)
    - Multimodal content with images (content array format)

    Args:
        request: The model request containing parts and system prompt.

    Returns:
        List of LangChain message objects ready for ainvoke/astream.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from ..types import ImagePart, TextPart

    messages: list[object] = []
    if request.system and not is_reasoning_model:
        messages.append(SystemMessage(content=request.system))

    # Check if we have images
    has_images = any(isinstance(p, ImagePart) for p in request.parts)

    system_prefix = f"{request.system}\n\n" if (request.system and is_reasoning_model) else ""

    if has_images:
        # Build multimodal content array for vision models
        content: list[dict[str, object]] = []

        # Prepend system message to first text block if reasoning model
        if system_prefix:
            content.append({"type": "text", "text": system_prefix.strip()})

        for part in request.parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if part.uri:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": part.uri},
                        }
                    )
                elif part.data_b64:
                    data_url = f"data:{part.mime};base64,{part.data_b64}"
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
        messages.append(HumanMessage(content=content))
    else:
        # Text-only: simple string content
        text = request.to_text_prompt()
        if system_prefix:
            text = system_prefix + text
        messages.append(HumanMessage(content=text))

    return messages


async def generate_asr_openai_compat(
    request: "ModelRequest",
    *,
    base_url: str,
    api_key: str | None,
    provider: str,
    whisper_model: str = "whisper-1",
) -> "ModelResponse":
    """Generate ASR (speech-to-text) using OpenAI-compatible /audio/transcriptions.

    This is a shared implementation for providers that support the OpenAI
    audio transcriptions endpoint (OpenAI, Groq, etc.).

    Args:
        request: Model request containing AudioPart.
        base_url: Base URL for the API (e.g., "https://api.openai.com/v1").
        api_key: API key for authentication.
        provider: Provider name for response metadata.
        whisper_model: Whisper model to use (e.g., "whisper-1", "whisper-large-v3").

    Returns:
        ModelResponse with transcribed text.

    Raises:
        ValueError: If no AudioPart found or AudioPart has no data.
        ImportError: If httpx is not installed.
    """
    from ..types import AudioPart, ModelResponse, ProviderInfo

    audio_parts = [p for p in request.parts if isinstance(p, AudioPart)]
    if not audio_parts:
        raise ValueError("ASR requires at least one AudioPart")

    part = audio_parts[0]
    if not (part.uri or part.data_b64):
        raise ValueError("AudioPart requires either uri or data_b64")

    try:
        import httpx
    except ImportError as e:
        raise ImportError("ASR requires `httpx`") from e

    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build multipart form
    files: dict[str, tuple[str, bytes, str]] = {}
    if part.uri:
        p = Path(part.uri)
        data = p.read_bytes()
        filename = p.name
        mime = part.mime or "application/octet-stream"
        files["file"] = (filename, data, mime)
    else:
        data = base64.b64decode(part.data_b64 or "")
        files["file"] = ("audio", data, (part.mime or "application/octet-stream"))

    form = {"model": whisper_model}
    async with httpx.AsyncClient(timeout=request.timeout_sec) as client:
        resp = await client.post(url, headers=headers, data=form, files=files)
        resp.raise_for_status()
        payload = resp.json()
        text = payload.get("text") if isinstance(payload, dict) else None

    return ModelResponse(
        text=str(text or ""),
        raw_provider=ProviderInfo(
            provider=provider,
            model_name=whisper_model,
            model_key=f"{provider}/{whisper_model}",
        ),
    )


__all__ = [
    "build_openai_messages",
    "generate_asr_openai_compat",
]
