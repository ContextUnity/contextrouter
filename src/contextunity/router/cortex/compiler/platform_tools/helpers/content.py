"""Shared LLM invocation helpers for router content platform tools."""

from __future__ import annotations

from contextunity.core.exceptions import PlatformServiceError

from contextunity.router.cortex.compiler.state_routing import read_state_input

from .contracts import PlatformState


def require_model_key(*, state: PlatformState, model_override: str, tool_binding: str) -> str:
    """Resolve model key from config override or runtime state."""
    model_key = model_override or str(read_state_input(state, "model_key", default=""))
    if not model_key:
        raise PlatformServiceError(
            message=f"{tool_binding} requires a model key",
            tool_binding=tool_binding,
        )
    return model_key


async def run_text_generation(
    *,
    state: PlatformState,
    tool_binding: str,
    model_override: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None = None,
) -> str:
    """Run single text generation call through model registry."""
    from contextunity.router.modules.models import model_registry
    from contextunity.router.modules.models.types import ModelRequest, TextPart

    model_key = require_model_key(
        state=state,
        model_override=model_override,
        tool_binding=tool_binding,
    )
    llm = model_registry.create_llm(model_key)

    if max_output_tokens is None:
        request = ModelRequest(
            parts=[TextPart(text=prompt)],
            temperature=temperature,
        )
    else:
        request = ModelRequest(
            parts=[TextPart(text=prompt)],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    response = await llm.generate(request)
    return response.text


__all__ = ["require_model_key", "run_text_generation"]
