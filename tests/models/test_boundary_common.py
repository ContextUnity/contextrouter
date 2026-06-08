"""Tests for shared SDK boundary helpers."""

from contextunity.router.modules.models.boundary_common import (
    build_kwargs,
    openai_stream_delta_text,
    resolve_json_object_mode,
    resolve_max_output_tokens,
    resolve_request_field,
)


def test_build_kwargs_drops_none_values() -> None:
    """Optional SDK params must be omitted, not sent as null."""
    assert build_kwargs(model="gpt-5-mini", max_tokens=None, temperature=0.2) == {
        "model": "gpt-5-mini",
        "temperature": 0.2,
    }


def test_resolve_request_field_prefers_request() -> None:
    """Node executor sets generation params on ModelRequest, not provider_config."""
    assert resolve_request_field(0.1, 0.7) == 0.1
    assert resolve_request_field(None, 512) == 512
    assert resolve_request_field(None, None) is None


def test_resolve_max_output_tokens_prefers_request() -> None:
    assert resolve_max_output_tokens(request_max_output_tokens=256, provider_max_tokens=512) == 256
    assert resolve_max_output_tokens(request_max_output_tokens=None, provider_max_tokens=512) == 512


def test_resolve_json_object_mode_prefers_request() -> None:
    """Node executor sets response_format on ModelRequest, not provider_config."""
    assert resolve_json_object_mode(
        request_response_format="json_object",
        provider_response_format=None,
    )
    assert not resolve_json_object_mode(
        request_response_format=None,
        provider_response_format="text",
    )


def test_openai_stream_delta_text_reads_chunk() -> None:
    """Stream chunk parsing works through duck-typed objects."""

    class _Delta:
        content = "hi"

    class _Choice:
        delta = _Delta()

    class _Chunk:
        choices = [_Choice()]

    assert openai_stream_delta_text(_Chunk()) == "hi"
