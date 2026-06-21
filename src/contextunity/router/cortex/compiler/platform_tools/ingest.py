"""Router File Download platform tool — generic file acquisition for LangGraph agents.

Registers 1 self-hosted tool: ``router_file_download``.

This tool downloads files via HTTP(S) with optional authentication.
It is NOT called by Commerce or Traverse — those use their own fetcher
infrastructure. This exists so LangGraph agents can download files
during orchestration (e.g., RAG ingestion, data collection).

Architecture:
    Executor(state, config) → reads URL from state → download → writes content to state

Security:
    - Requires ``router:execute`` scope
    - Config schema: frozen=True, extra=forbid
    - Credentials resolved via state (manifest config), NOT hardcoded
    - Size limits enforced

Uses stdlib urllib only — zero external dependencies.
"""

from __future__ import annotations

import base64
import http.client
import time
import urllib.parse
from email.message import Message
from typing import TYPE_CHECKING, ClassVar, Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError, SecurityError
from contextunity.core.security import validate_safe_url
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..platform_registry import PlatformToolRegistry

from .helpers.contracts import PlatformResult, PlatformState
from .helpers.registration import ToolRegistrationSpec, register_tool_specs
from .helpers.state import as_text

logger = get_contextunit_logger(__name__)


def _download_http_content(
    *,
    url: str,
    headers: dict[str, str],
    timeout: int,
    max_bytes: int,
    display_url: str | None = None,
) -> tuple[bytes, str, str | None]:
    """Download URL content through typed stdlib HTTP clients."""
    try:
        safe_url = validate_safe_url(url)
    except SecurityError as exc:
        raise PlatformServiceError(
            message="router_file_download: unsafe URL rejected",
            tool_binding="router_file_download",
        ) from exc

    log_url = display_url or safe_url
    parsed = urllib.parse.urlsplit(safe_url)
    host = parsed.hostname
    if not host:
        raise PlatformServiceError(
            message=f"router_file_download: missing host in URL: {log_url}",
            tool_binding="router_file_download",
        )

    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            host,
            port=parsed.port,
            timeout=timeout,
        )
    elif parsed.scheme == "http":
        connection = http.client.HTTPConnection(
            host,
            port=parsed.port,
            timeout=timeout,
        )
    else:
        raise PlatformServiceError(
            message=f"router_file_download: invalid URL scheme: {log_url}",
            tool_binding="router_file_download",
        )

    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        if response.status >= 400:
            raise PlatformServiceError(
                message=f"router_file_download: HTTP {response.status} for {log_url}",
                tool_binding="router_file_download",
            )
        data = response.read(max_bytes + 1)
        content_type = response.getheader("Content-Type", "")
        content_disposition = response.getheader("Content-Disposition")
        filename: str | None = None
        if content_disposition:
            msg = Message()
            msg.add_header("Content-Disposition", content_disposition)
            filename_raw = msg.get_param("filename", header="Content-Disposition")
            filename = str(filename_raw) if filename_raw is not None else None
        return data, content_type, filename
    finally:
        connection.close()


def _redact_url_query_params(url: str, params: set[str]) -> str:
    """Redact sensitive query params before logging or returning metadata."""
    if not params:
        return url
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return url
    sensitive = {param.lower() for param in params}
    query = [
        (key, "REDACTED" if key.lower() in sensitive else value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


# ── Config Schema ───────────────────────────────────────────────────


class FileDownloadConfig(BaseModel, frozen=True):
    """Config for router_file_download tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    url_key: str = Field(
        default="download_url",
        description="State key holding the URL to download",
    )
    output_key: str = Field(
        default="downloaded_content",
        description="State key for output (base64-encoded bytes or text)",
    )
    auth_mode: Literal["none", "basic", "api_key"] = Field(
        default="none",
        description="Authentication mode",
    )
    username_key: str = Field(
        default="download_username",
        description="State key for username (basic auth)",
    )
    password_key: str = Field(
        default="download_password",
        description="State key for password (basic auth)",
    )
    api_key_key: str = Field(
        default="download_api_key",
        description="State key for API key",
    )
    api_key_param: str = Field(
        default="key",
        description="Query parameter name for API key",
    )
    timeout: int = Field(default=30, ge=5, le=300)
    max_size_mb: int = Field(default=50, ge=1, le=500)
    retries: int = Field(default=3, ge=1, le=10)
    output_format: Literal["base64", "text"] = Field(
        default="text",
        description="Output format: 'text' for UTF-8 decoded, 'base64' for binary",
    )


# ── Executor ────────────────────────────────────────────────────────


async def _router_file_download_executor(
    state: PlatformState, config: FileDownloadConfig
) -> PlatformResult:
    """Download a file from a URL with optional authentication.

    Reads URL from state[config.url_key], downloads content,
    and writes result to state[config.output_key].
    """
    try:
        url = as_text(state.get(config.url_key, ""))
        if not url:
            raise PlatformServiceError(
                message=(f"router_file_download: no URL found in state['{config.url_key}']"),
                tool_binding="router_file_download",
            )

        if not url.startswith(("http://", "https://")):
            raise PlatformServiceError(
                message=f"router_file_download: invalid URL scheme: {url}",
                tool_binding="router_file_download",
            )

        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (compatible; contextunity-router/1.0)",
        }
        sensitive_query_params: set[str] = set()

        if config.auth_mode == "basic":
            username = as_text(state.get(config.username_key, ""))
            password = as_text(state.get(config.password_key, ""))
            if username and password:
                credentials = f"{username}:{password}"
                encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
                headers["Authorization"] = f"Basic {encoded}"

        elif config.auth_mode == "api_key":
            api_key = as_text(state.get(config.api_key_key, ""))
            if api_key:
                parsed = urllib.parse.urlparse(url)
                query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                query[config.api_key_param] = [api_key]
                sensitive_query_params.add(config.api_key_param)
                url = urllib.parse.urlunparse(
                    parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
                )

        max_bytes = config.max_size_mb * 1024 * 1024
        display_url = _redact_url_query_params(url, sensitive_query_params)

        # Download with retry
        last_error: Exception | None = None
        for attempt in range(1, config.retries + 1):
            try:
                data, content_type, filename = _download_http_content(
                    url=url,
                    headers=headers,
                    timeout=config.timeout,
                    max_bytes=max_bytes,
                    display_url=display_url,
                )
                if len(data) > max_bytes:
                    raise PlatformServiceError(
                        message=(f"File exceeds {config.max_size_mb}MB limit"),
                        tool_binding="router_file_download",
                    )

                if config.output_format == "base64":
                    content = base64.b64encode(data).decode("ascii")
                else:
                    content = data.decode("utf-8", errors="replace")

                logger.info("Downloaded %d bytes from %s", len(data), display_url)

                return {
                    config.output_key: content,
                    "download_metadata": {
                        "url": display_url,
                        "size_bytes": len(data),
                        "content_type": content_type,
                        "filename": filename,
                    },
                }

            except PlatformServiceError:
                raise
            except (http.client.HTTPException, OSError, TimeoutError) as e:
                last_error = e
                if attempt < config.retries:
                    time.sleep(min(attempt * 2, 10))
                    continue

        raise PlatformServiceError(
            message=(
                f"Failed to download {display_url} after {config.retries} attempts: {last_error}"
            ),
            tool_binding="router_file_download",
        )

    except PlatformServiceError:
        raise
    except Exception as exc:  # wraps-to-domain: re-raises as typed exception
        raise PlatformServiceError(
            message="router_file_download execution failed",
            tool_binding="router_file_download",
        ) from exc


# ── Registration ────────────────────────────────────────────────────


def register_ingest_tools(registry: PlatformToolRegistry) -> None:
    """Register file download tools into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="router_file_download",
                executor=_router_file_download_executor,
                config_schema=FileDownloadConfig,
                required_scopes=["router:execute"],
            )
        ],
    )


__all__ = [
    "register_ingest_tools",
    "FileDownloadConfig",
]
