"""Typed Protocol boundary for langchain-google-vertexai and Google auth."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from contextunity.core.exceptions import ConfigurationError
from contextunity.core.types import JsonDict, is_json_dict, is_object_list, is_object_tuple
from langchain_core.messages import BaseMessage

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@runtime_checkable
class GoogleCredentialsProtocol(Protocol):
    def refresh(self, request: object) -> None: ...


@runtime_checkable
class VertexRawModel(Protocol):
    async def ainvoke(self, input: list[BaseMessage]) -> object: ...

    def astream(self, input: list[BaseMessage]) -> AsyncIterator[object]: ...

    def bind(self, **kwargs: object) -> object: ...

    def get_num_tokens(self, text: str) -> int: ...

    @property
    def model_name(self) -> str: ...


def load_credentials_from_file(path: str) -> GoogleCredentialsProtocol:
    """Load service-account credentials from *path*."""
    oauth2 = importlib.import_module("google.oauth2.service_account")
    credentials_cls: object = getattr(oauth2, "Credentials", None)
    from_file_fn: object = getattr(credentials_cls, "from_service_account_file", None)
    if not callable(from_file_fn):
        raise ConfigurationError("google.oauth2.service_account.Credentials is unavailable")
    creds: object = from_file_fn(path, scopes=[_CLOUD_PLATFORM_SCOPE])
    if not isinstance(creds, GoogleCredentialsProtocol):
        raise ConfigurationError("Service account credentials are incompatible")
    return creds


def load_adc_credentials() -> GoogleCredentialsProtocol | None:
    """Load Application Default Credentials when available."""
    google_auth = importlib.import_module("google.auth")
    default_fn: object = getattr(google_auth, "default", None)
    if not callable(default_fn):
        return None
    result: object = default_fn(scopes=[_CLOUD_PLATFORM_SCOPE])
    if not is_object_tuple(result) or len(result) < 1:
        return None
    creds: object = result[0]
    if isinstance(creds, GoogleCredentialsProtocol):
        return creds
    return None


def refresh_credentials(creds: GoogleCredentialsProtocol) -> None:
    """Refresh *creds* using the Google auth transport request helper."""
    transport = importlib.import_module("google.auth.transport.requests")
    request_factory: object = getattr(transport, "Request", None)
    if not callable(request_factory):
        raise ConfigurationError("google.auth.transport.requests.Request is unavailable")
    request_obj: object = request_factory()
    creds.refresh(request_obj)


def load_chat_vertex_ai(**kwargs: object) -> VertexRawModel:
    """Construct ``ChatVertexAI`` when the optional dependency is installed."""
    vertex_mod = importlib.import_module("langchain_google_vertexai")
    chat_cls: object = getattr(vertex_mod, "ChatVertexAI", None)
    if not callable(chat_cls):
        raise ConfigurationError("langchain_google_vertexai.ChatVertexAI is unavailable")
    model: object = chat_cls(**kwargs)
    if not isinstance(model, VertexRawModel):
        raise ConfigurationError("ChatVertexAI returned an incompatible model instance")
    return model


def vertex_ai_message_text(msg: object) -> str:
    """Extract assistant text from a LangChain AIMessage-like object."""
    content: object = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def vertex_chunk_content_text(raw_content: object) -> str:
    """Normalize streaming chunk content to plain text."""
    if isinstance(raw_content, str):
        return raw_content
    if is_object_list(raw_content):
        parts: list[str] = []
        for part_item in raw_content:
            if is_json_dict(part_item):
                if part_item.get("type") == "text":
                    text_val = part_item.get("text")
                    parts.append(str(text_val) if text_val is not None else "")
            elif isinstance(part_item, str):
                parts.append(part_item)
        return "".join(parts)
    return str(raw_content) if raw_content else ""


def vertex_usage_metadata_mapping(um: object) -> JsonDict:
    """Normalize usage metadata objects to a token-count mapping."""
    if is_json_dict(um):
        return um
    return {
        "input_tokens": getattr(um, "input_tokens", None),
        "output_tokens": getattr(um, "output_tokens", None),
        "total_tokens": getattr(um, "total_tokens", None),
    }


def vertex_response_metadata(msg: object) -> JsonDict | None:
    """Read ``response_metadata`` from a LangChain message when present."""
    rm: object = getattr(msg, "response_metadata", None)
    if is_json_dict(rm):
        return rm
    return None


__all__ = [
    "GoogleCredentialsProtocol",
    "VertexRawModel",
    "load_adc_credentials",
    "load_chat_vertex_ai",
    "load_credentials_from_file",
    "refresh_credentials",
    "vertex_ai_message_text",
    "vertex_chunk_content_text",
    "vertex_response_metadata",
    "vertex_usage_metadata_mapping",
]
