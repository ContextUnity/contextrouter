"""Security behavior for router_file_download platform tool."""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import PlatformServiceError

from contextunity.router.cortex.compiler.platform_tools import ingest
from contextunity.router.cortex.compiler.platform_tools.ingest import (
    FileDownloadConfig,
    _download_http_content,
    _router_file_download_executor,
)


def test_file_download_rejects_metadata_endpoint() -> None:
    with pytest.raises(PlatformServiceError, match="unsafe URL"):
        _download_http_content(
            url="http://169.254.169.254/latest/meta-data/",
            headers={},
            timeout=5,
            max_bytes=1024,
        )


@pytest.mark.asyncio
async def test_file_download_redacts_api_key_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_download_http_content(
        *,
        url: str,
        headers: dict[str, str],
        timeout: int,
        max_bytes: int,
        display_url: str | None,
    ) -> tuple[bytes, str, str | None]:
        captured["url"] = url
        captured["display_url"] = display_url or ""
        captured["headers_user_agent"] = headers["User-Agent"]
        assert timeout == 5
        assert max_bytes == 1024 * 1024
        return b"payload", "text/plain", "payload.txt"

    monkeypatch.setattr(ingest, "_download_http_content", fake_download_http_content)

    result = await _router_file_download_executor(
        {
            "download_url": "https://example.com/file.txt?keep=1",
            "download_api_key": "secret-token",
        },
        FileDownloadConfig(
            auth_mode="api_key",
            api_key_param="token",
            timeout=5,
            max_size_mb=1,
            retries=1,
        ),
    )

    assert "secret-token" in captured["url"]
    assert "secret-token" not in captured["display_url"]
    assert "token=REDACTED" in captured["display_url"]
    assert result["download_metadata"]["url"] == captured["display_url"]
    assert "secret-token" not in repr(result)
